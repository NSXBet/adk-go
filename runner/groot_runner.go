package runner

import (
	"context"
	"fmt"
	"iter"
	"log"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/runner/internal"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/genai"
)

type GRootRunnerConfig struct {
	GRootEndpoint  string
	GRootAPIKey    string
	GRootSessionID string

	AppName        string
	RootAgent      agent.Agent
	SessionService sessionservice.Service
}

type GRootRunner struct {
	cfg *GRootRunnerConfig

	parents parentmap.Map
	session *internal.Session
}

func NewGRootRunner(cfg *GRootRunnerConfig) (*GRootRunner, error) {
	client, err := internal.NewClient(cfg.GRootEndpoint, cfg.GRootAPIKey)
	if err != nil {
		return nil, err
	}

	if cfg.GRootSessionID == "" {
		cfg.GRootSessionID = uuid.NewString()
	}
	if cfg.SessionService == nil {
		cfg.SessionService = sessionservice.Mem()
	}
	sess, err := client.OpenSession(cfg.GRootSessionID)
	if err != nil {
		return nil, err
	}
	return &GRootRunner{
		cfg:     cfg,
		session: sess,
	}, nil
}

func (r *GRootRunner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *RunConfig) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// TODO: Replace with Get.
		session, err := r.cfg.SessionService.Create(ctx, &sessionservice.CreateRequest{
			AppName:   r.cfg.AppName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			yield(nil, err)
			return
		}

		agentToRun, err := r.findAgentToRun(session)
		if err != nil {
			yield(nil, err)
			return
		}

		if cfg != nil && cfg.SupportCFC {
			if err := r.setupCFC(agentToRun); err != nil {
				yield(nil, fmt.Errorf("failed to setup CFC: %w", err))
				return
			}
		}

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
		})

		ctx := agent.NewContext(ctx, agentToRun, msg, &mutableSession{
			service:       r.cfg.SessionService,
			storedSession: session,
		}, "")

		if err := r.appendMessageToSession(ctx, session, msg); err != nil {
			yield(nil, err)
			return
		}

		input := uuid.NewString()
		output := uuid.NewString()
		shadow, status, err := r.session.NewADKShadow(agentToRun.Name(), input, output)
		if err != nil {
			if !yield(nil, err) {
				return
			}
		}
		if status == internal.ShadowStatusCompleted {
			// TODO: Retrieve the previous.
		}

		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				if event.LLMResponse != nil {
					if !yield(event, err) {
						return
					}
				}
				continue
			}

			// TODO: Write the entire event.
			for _, part := range event.LLMResponse.Content.Parts {
				if err := shadow.WriteFrame(output, internal.ChunkFromPart(part), event.LLMResponse.Partial); err != nil {
					if !yield(nil, err) {
						return
					}
				}
			}

			// only commit non-partial event to a session service
			if !(event.LLMResponse != nil && event.LLMResponse.Partial) {
				// TODO: update session state & delta
				if err := r.cfg.SessionService.AppendEvent(ctx, session, event); err != nil {
					yield(nil, fmt.Errorf("failed to add event to session: %w", err))
					return
				}
			}

			if !yield(event, nil) {
				return
			}
		}
	}
}

// findAgentToRun returns the agent that should handle the next request based on
// session history.
func (r *GRootRunner) findAgentToRun(session sessionservice.StoredSession) (agent.Agent, error) {
	events := session.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)

		// TODO: findMatchingFunctionCall.

		if event.Author == "user" {
			continue
		}

		subAgent := findAgent(r.cfg.RootAgent, event.Author)
		// Agent not found, continue looking for the other event.
		if subAgent == nil {
			log.Printf("Event from an unknown agent: %s, event id: %s", event.Author, event.ID)
			continue
		}

		if r.isTransferableAcrossAgentTree(subAgent) {
			return subAgent, nil
		}
	}

	// Falls back to root agent if no suitable agents are found in the session.
	return r.cfg.RootAgent, nil
}

// checks if the agent and its parent chain allow transfer up the tree.
func (r *GRootRunner) isTransferableAcrossAgentTree(agentToRun agent.Agent) bool {
	for curAgent := agentToRun; curAgent != nil; curAgent = r.parents[curAgent.Name()] {
		llmAgent, ok := agentToRun.(llminternal.Agent)
		if !ok {
			return false
		}
		if llminternal.Reveal(llmAgent).DisallowTransferToParent {
			return false
		}
	}

	return true
}

func (r *GRootRunner) setupCFC(curAgent agent.Agent) error {
	llmAgent, ok := curAgent.(llminternal.Agent)
	if !ok {
		return fmt.Errorf("agent %v is not an LLMAgent", curAgent.Name())
	}

	model := llminternal.Reveal(llmAgent).Model

	if model == nil {
		return fmt.Errorf("LLMAgent has no model")
	}

	if !strings.HasPrefix(model.Name(), "gemini-2") {
		return fmt.Errorf("CFC is not supported for model: %v", model.Name())
	}

	// TODO: handle CFC setup for LLMAgent, e.g. setting code_executor
	return nil
}

func (r *GRootRunner) appendMessageToSession(ctx agent.Context, storedSession sessionservice.StoredSession, msg *genai.Content) error {
	if msg == nil {
		return nil
	}
	event := session.NewEvent(ctx.InvocationID())

	event.Author = "user"
	event.LLMResponse = &llm.Response{
		Content: msg,
	}

	if err := r.cfg.SessionService.AppendEvent(ctx, storedSession, event); err != nil {
		return fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return nil
}
