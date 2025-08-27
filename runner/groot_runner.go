package runner

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log"
	"os"
	"strings"
	"sync"

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

const defaultEventLog = "adk_runner.log"

type GRootRunnerConfig struct {
	GRootEndpoint  string
	GRootAPIKey    string
	GRootSessionID string

	EventLog string

	AppName        string
	RootAgent      agent.Agent
	SessionService sessionservice.Service
}

type GRootRunner struct {
	cfg *GRootRunnerConfig

	parents  parentmap.Map
	registry *internal.Registry
	eventLog *EventLog
}

func NewGRootRunner(cfg *GRootRunnerConfig) (*GRootRunner, error) {
	return newGRootRunner(cfg, nil)
}

func newGRootRunner(cfg *GRootRunnerConfig, elog *EventLog) (*GRootRunner, error) {
	if cfg.SessionService == nil {
		cfg.SessionService = sessionservice.Mem()
	}
	client, err := internal.NewClient(cfg.GRootEndpoint, cfg.GRootAPIKey)
	if err != nil {
		return nil, err
	}
	if elog == nil {
		if cfg.EventLog == "" {
			// TODO: Event log should be per app, and per user.
			// For this demo, we will use a single log file
			// assuming there is a single app and a single user.
			cfg.EventLog = defaultEventLog
		}
		elog, err = openEventLog(cfg.AppName, cfg.EventLog, client, cfg.GRootSessionID, true)
		if err != nil {
			return nil, err
		}
	}
	return &GRootRunner{
		cfg:      cfg,
		eventLog: elog,
		registry: internal.NewRegistry(cfg.RootAgent),
	}, nil
}

func (r *GRootRunner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *RunConfig) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		session, err := r.cfg.SessionService.Get(ctx, &sessionservice.GetRequest{
			ID: session.ID{
				AppName:   r.cfg.AppName,
				UserID:    userID,
				SessionID: sessionID,
			},
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

		input := uuid.NewString()
		output := uuid.NewString()
		branch := input + ":" + output

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
		})

		ctx := agent.NewContext(ctx, agentToRun, msg, &mutableSession{
			service:       r.cfg.SessionService,
			storedSession: session,
		}, branch)

		if err := r.appendMessageToSession(ctx, session, msg); err != nil {
			yield(nil, err)
			return
		}

		if err := r.eventLog.LogActivity(
			userID, "agent_start", r.registry.AgentFullname(agentToRun), input, output); err != nil {
			if !yield(nil, err) {
				return
			}
		}
		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				return
			}
			if err := r.cfg.SessionService.AppendEvent(ctx, session, event); err != nil {
				yield(nil, fmt.Errorf("failed to add event to session: %w", err))
				return
			}
			if err := r.eventLog.LogEvent(userID, output, event); err != nil {
				if !yield(nil, err) {
					return
				}
			}
			if !yield(event, nil) {
				return
			}
		}
		if err := r.eventLog.LogActivity(
			userID, "agent_end", r.registry.AgentFullname(agentToRun), input, output); err != nil {
			if !yield(nil, err) {
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

type EventLog struct {
	appName string
	logFile *os.File
	session *internal.Session
}

func openEventLog(appName string, filename string, client *internal.Client, sid string, create bool) (*EventLog, error) {
	sess, err := client.OpenSession(sid)
	if err != nil {
		return nil, err
	}

	perm := os.O_RDWR | os.O_APPEND
	if create {
		perm = os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	}

	file, err := os.OpenFile(filename, perm, 0644)
	if err != nil {
		return nil, err
	}
	return &EventLog{
		appName: appName,
		logFile: file,
		session: sess,
	}, nil
}

type ActivityEvent struct {
	Kind   string `json:"kind,omitempty"`
	Name   string `json:"name,omitempty"`
	Input  string `json:"input,omitempty"`
	Output string `json:"output,omitempty"`
}

type StreamEvent struct {
	Kind     string `json:"kind,omitempty"`
	StreamID string `json:"stream_id,omitempty"`
}

func (e *EventLog) LogEvent(userID string, id string, event *session.Event) error {
	if event.LLMResponse == nil || event.LLMResponse.Content == nil {
		return nil
	}
	out, err := json.Marshal(event)
	if err != nil {
		return err
	}
	if err := e.session.WriteFrame(id, &internal.Chunk{
		MIMEType: "application/json",
		Data:     out,
	}, event.Partial); err != nil {
		return err
	}
	if event.Partial {
		// Only log the event when all writes are finished.
		return nil
	}
	return e.logJSON(&StreamEvent{
		Kind:     "stream",
		StreamID: id,
	})
}

func (e *EventLog) LogActivity(userID string, kind string, name, input, output string) error {
	return e.logJSON(&ActivityEvent{
		Kind:   kind,
		Name:   name,
		Input:  input,
		Output: output,
	})
}

func (e *EventLog) logJSON(v any) error {
	out, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(e.logFile, "%s\n", out)
	if err != nil {
		return err
	}
	return e.logFile.Sync()
}

func ResumerEventLog(appName, filename string, client *internal.Client, sessionID string) (*EventLog, error) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}
	sess, err := client.OpenSession(sessionID)
	if err != nil {
		return nil, err
	}
	return &EventLog{
		appName: appName,
		session: sess,
		logFile: file,
	}, nil
}

type ResumerGRootRunner struct {
	cfg      *GRootRunnerConfig
	eventLog *EventLog
	delegate *GRootRunner
	registry *internal.Registry

	replayOnce sync.Once
}

func NewResumerGRootRunner(cfg *GRootRunnerConfig) (*ResumerGRootRunner, error) {
	if cfg.SessionService == nil {
		cfg.SessionService = sessionservice.Mem()
	}
	if cfg.EventLog == "" {
		return nil, errors.New("event log is not set; required to resume")
	}
	client, err := internal.NewClient(cfg.GRootEndpoint, cfg.GRootAPIKey)
	if err != nil {
		return nil, err
	}
	elog, err := openEventLog(cfg.AppName, cfg.EventLog, client, cfg.GRootSessionID, false)
	if err != nil {
		return nil, err
	}
	delegate, err := newGRootRunner(cfg, elog)
	if err != nil {
		return nil, err
	}
	return &ResumerGRootRunner{
		cfg:      cfg,
		eventLog: elog,
		delegate: delegate, // once resumption is done, delegate continues
		registry: internal.NewRegistry(cfg.RootAgent),
	}, nil
}

func (r *ResumerGRootRunner) replay(ctx context.Context, session sessionservice.StoredSession, userID, sessionID string) error {
	// TODO: Replay needs to be played for the given user ID.
	waiting := make(map[string]*ActivityEvent)
	scanner := bufio.NewScanner(r.eventLog.logFile)
	for scanner.Scan() {
		stream, act, err := r.parse(scanner.Bytes())
		if err != nil {
			return err
		}
		if stream != nil {
			if err := r.replayStream(ctx, session, stream); err != nil {
				return err
			}
		}
		if act != nil {
			// Figure out what to do with the activity.
			key := act.Name + ":" + act.Input + ":" + act.Output
			if act.Kind == "agent_start" {
				waiting[key] = act
			}
			if act.Kind == "agent_end" {
				delete(waiting, key)
			}
		}
	}
	// TODO(jbd): Does order matter?
	// Allow concurrency in the future.
	for _, act := range waiting {
		if err := r.replayAgent(ctx, act.Name, act.Input, act.Output); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func (r *ResumerGRootRunner) replayAgent(ctx context.Context, agent, input, output string) error {
	a := r.registry.Resolve(agent)
	if a == nil {
		return fmt.Errorf("agent %q not found", agent)
	}
	panic("not implemented")
	// TODO: Check if input is available.
	// Write output to output.
	// TODO: Log agent_end.
}

func (r *ResumerGRootRunner) replayStream(ctx context.Context, sess sessionservice.StoredSession, stream *StreamEvent) error {
	chunks, err := r.eventLog.session.ReadAll(stream.StreamID)
	if err != nil {
		return err
	}
	for _, chunk := range chunks {
		var event session.Event
		if chunk == nil || chunk.Data == nil {
			continue
		}
		if err := json.Unmarshal(chunk.Data, &event); err != nil {
			return err
		}
		if err := r.cfg.SessionService.AppendEvent(ctx, sess, &event); err != nil {
			return err
		}
	}
	return nil
}

func (r *ResumerGRootRunner) parse(b []byte) (*StreamEvent, *ActivityEvent, error) {
	var event StreamEvent
	var activity ActivityEvent
	if err := json.Unmarshal(b, &event); err != nil {
		return nil, nil, err
	}
	if event.Kind == "stream" {
		return &event, nil, nil
	}
	if err := json.Unmarshal(b, &activity); err != nil {
		return nil, nil, err
	}
	return nil, &activity, nil
}

func (r *ResumerGRootRunner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *RunConfig) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		session, err := r.cfg.SessionService.Get(ctx, &sessionservice.GetRequest{
			ID: session.ID{
				AppName:   r.cfg.AppName,
				UserID:    userID,
				SessionID: sessionID,
			},
		})
		if err != nil {
			yield(nil, err)
			return
		}
		// TODO(jbd): Replayer event log will be per application and user.
		// Resumer should support multitenancy. We skip this effort in the prototype.
		r.replayOnce.Do(func() {
			if err := r.replay(ctx, session, userID, sessionID); err != nil {
				yield(nil, err)
				return
			}
		})
		for event, err := range r.delegate.Run(ctx, userID, sessionID, msg, cfg) {
			if !yield(event, err) {
				return
			}
		}
	}
}
