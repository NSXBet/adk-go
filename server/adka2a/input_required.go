// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package adka2a

import (
	"fmt"
	"slices"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

type inputRequiredProcessor struct {
	execCtx *a2asrv.ExecutorContext
	event   *a2a.TaskStatusUpdateEvent
	// handles possible duplication in partial and non-partial events
	addedParts []*genai.Part
}

func newInputRequiredProcessor(execCtx *a2asrv.ExecutorContext) *inputRequiredProcessor {
	return &inputRequiredProcessor{execCtx: execCtx}
}

// process handles long-running function tool calls by accumulating them for the final task status update.
// If a part was incorporated into the final task status update the original event is modified to not include it,
// so that parts are not duplicated in the response.
func (p *inputRequiredProcessor) process(event *session.Event) (*session.Event, error) {
	resp := event.LLMResponse
	if resp.Content == nil {
		return event, nil
	}

	var longRunningCallIDs []string
	var inputRequiredParts []*genai.Part
	var remainingParts []*genai.Part
	for _, part := range resp.Content.Parts {
		callID := ""
		if part.FunctionCall != nil && slices.Contains(event.LongRunningToolIDs, part.FunctionCall.ID) {
			callID = part.FunctionCall.ID
		}
		if p.isLongRunningResponse(event, part) {
			callID = part.FunctionResponse.ID
		}
		if callID == "" {
			remainingParts = append(remainingParts, part)
			continue
		}
		added := slices.ContainsFunc(p.addedParts, func(p *genai.Part) bool {
			if part.FunctionCall != nil && p.FunctionCall != nil && part.FunctionCall.ID == p.FunctionCall.ID {
				return true
			}
			return part.FunctionResponse != nil && p.FunctionResponse != nil && part.FunctionResponse.ID == p.FunctionResponse.ID
		})
		if added {
			continue
		}
		p.addedParts = append(p.addedParts, part)
		inputRequiredParts = append(inputRequiredParts, part)
		longRunningCallIDs = append(longRunningCallIDs, callID)
	}

	if len(inputRequiredParts) > 0 {
		a2aParts, err := ToA2AParts(inputRequiredParts, longRunningCallIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to convert input required parts to A2A parts: %w", err)
		}

		if p.event != nil {
			p.event.Status.Message.Parts = append(p.event.Status.Message.Parts, a2aParts...)
		} else {
			msg := a2a.NewMessage(a2a.MessageRoleAgent, a2aParts...)
			ev := a2a.NewStatusUpdateEvent(p.execCtx, a2a.TaskStateInputRequired, msg)
			p.event = ev
		}
	}

	if len(remainingParts) == len(resp.Content.Parts) {
		return event, nil
	}

	modifiedEvent := *event
	newContent := *resp.Content
	newContent.Parts = remainingParts
	modifiedEvent.LLMResponse.Content = &newContent

	return &modifiedEvent, nil
}

func (p *inputRequiredProcessor) isLongRunningResponse(event *session.Event, part *genai.Part) bool {
	if part.FunctionResponse == nil {
		return false
	}
	id := part.FunctionResponse.ID
	if slices.Contains(event.LongRunningToolIDs, id) {
		return true
	}
	if p.event == nil {
		return false
	}
	for _, msgPart := range p.event.Status.Message.Parts {
		if msgPart.Data() != nil {
			if typeVal, ok := msgPart.Metadata[a2aDataPartMetaTypeKey]; ok && typeVal == a2aDataPartTypeFunctionCall {
				if data, ok := msgPart.Data().(map[string]any); ok {
					if callID, ok := data["id"].(string); ok && callID == id {
						return true
					}
				}
			}
		}
	}
	return false
}

// handleInputRequired checks if the input message contains responses to all function calls
// that happened during the previous invocation and were recorded in the Task input-required state message.
// If a non-nil event is returned the invoking code needs to use the event as the result of the execution
func handleInputRequired(execCtx *a2asrv.ExecutorContext, content *genai.Content) (*a2a.TaskStatusUpdateEvent, error) {
	if execCtx.StoredTask == nil {
		return nil, nil
	}
	task, statusMsg := execCtx.StoredTask, execCtx.StoredTask.Status.Message
	if task.Status.State != a2a.TaskStateInputRequired || statusMsg == nil {
		return nil, nil
	}

	taskParts, err := ToGenAIParts(statusMsg.Parts)
	if err != nil {
		return nil, fmt.Errorf("failed to parse task status message: %w", err)
	}

	for _, statusPart := range taskParts {
		if statusPart.FunctionCall == nil {
			continue
		}
		hasMatchingResponse := slices.ContainsFunc(content.Parts, func(p *genai.Part) bool {
			return p.FunctionResponse != nil && p.FunctionResponse.ID == statusPart.FunctionCall.ID
		})
		if !hasMatchingResponse {
			parts := makeInputMissingErrorMessage(statusMsg.Parts, statusPart.FunctionCall.ID)
			msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, execCtx.StoredTask, parts...)
			event := a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateInputRequired, msg)
			return event, nil
		}
	}
	return nil, nil
}

func makeInputMissingErrorMessage(inputRequiredParts []*a2a.Part, callID string) []*a2a.Part {
	errPart := a2a.NewTextPart(fmt.Sprintf("no input provided for function call ID %q", callID))
	errPart.Metadata = map[string]any{"validation_error": true}
	var preservedParts []*a2a.Part
	for _, p := range inputRequiredParts {
		if p.Metadata != nil {
			if v, ok := p.Metadata["validation_error"].(bool); ok && v {
				continue
			}
		}
		preservedParts = append(preservedParts, p)
	}
	return append(preservedParts, errPart)
}
