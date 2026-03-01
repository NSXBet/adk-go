// Copyright 2025 Google LLC
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
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/a2aproject/a2a-go/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/converters"
)

var (
	a2aDataPartMetaTypeKey        = ToA2AMetaKey("type")
	a2aDataPartMetaLongRunningKey = ToA2AMetaKey("is_long_running")
)

const (
	a2aDataPartTypeFunctionCall       = "function_call"
	a2aDataPartTypeFunctionResponse   = "function_response"
	a2aDataPartTypeCodeExecResult     = "code_execution_result"
	a2aDataPartTypeCodeExecutableCode = "executable_code"
)

// IsPartial takes metadata of an A2A object (eg. a2a.Part, a2a.Artifact) and returs true if
// it was marked as partial based on the ADK partial flag set on the original ADK object.
func IsPartial(meta map[string]any) bool {
	if meta == nil {
		return false
	}
	isPartial, _ := meta[metadataPartialKey].(bool)
	return isPartial
}

// IsPartialFlagSet takes metadata of an A2A object (eg. a2a.Part, a2a.Artifact) and returs true if
// the ADK partial flag was set on it.
func IsPartialFlagSet(meta map[string]any) bool {
	if meta == nil {
		return false
	}
	_, isSet := meta[metadataPartialKey].(bool)
	return isSet
}

// ToA2APart converts the provided genai part to A2A equivalent. Long running tool IDs are used for attaching metadata to
// the relevant data parts.
func ToA2APart(part *genai.Part, longRunningToolIDs []string) (*a2a.Part, error) {
	parts, err := ToA2AParts([]*genai.Part{part}, longRunningToolIDs)
	if err != nil {
		return nil, err
	}
	return parts[0], nil
}

// ToA2AParts converts the provided genai parts to A2A equivalents. Long running tool IDs are used for attaching metadata to
// the relevant data parts.
func ToA2AParts(parts []*genai.Part, longRunningToolIDs []string) ([]*a2a.Part, error) {
	result := make([]*a2a.Part, len(parts))
	for i, part := range parts {
		if part.Text != "" {
			r := a2a.NewTextPart(part.Text)
			if part.Thought {
				r.Metadata = map[string]any{ToA2AMetaKey("thought"): true}
			}
			result[i] = r
		} else if part.InlineData != nil || part.FileData != nil {
			r, err := toA2AFilePart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else {
			r, err := toA2ADataPart(part, longRunningToolIDs)
			if err != nil {
				return nil, err
			}
			result[i] = r
		}
	}
	return result, nil
}

func updatePartsMetadata(parts []*a2a.Part, update map[string]any) {
	for _, part := range parts {
		if part.Metadata == nil {
			part.Metadata = make(map[string]any)
		}
		maps.Copy(part.Metadata, update)
	}
}

func toA2AFilePart(v *genai.Part) (*a2a.Part, error) {
	if v == nil || (v.FileData == nil && v.InlineData == nil) {
		return nil, fmt.Errorf("not a file part: %v", v)
	}

	if v.FileData != nil {
		p := a2a.NewFileURLPart(a2a.URL(v.FileData.FileURI), v.FileData.MIMEType)
		p.Filename = v.FileData.DisplayName
		return p, nil
	}

	p := a2a.NewRawPart(v.InlineData.Data)
	p.MediaType = v.InlineData.MIMEType
	p.Filename = v.InlineData.DisplayName

	if v.VideoMetadata != nil {
		data, err := converters.ToMapStructure(v.VideoMetadata)
		if err != nil {
			return nil, err
		}
		p.Metadata = map[string]any{"video_metadata": data}
	}

	return p, nil
}

func toA2ADataPart(part *genai.Part, longRunningToolIDs []string) (*a2a.Part, error) {
	if part.CodeExecutionResult != nil {
		data, err := converters.ToMapStructure(part.CodeExecutionResult)
		if err != nil {
			return nil, err
		}
		p := a2a.NewDataPart(data)
		p.Metadata = map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecResult}
		return p, nil
	}

	if part.FunctionResponse != nil {
		data, err := converters.ToMapStructure(part.FunctionResponse)
		if err != nil {
			return nil, err
		}
		p := a2a.NewDataPart(data)
		p.Metadata = map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionResponse}
		return p, nil
	}

	if part.ExecutableCode != nil {
		data, err := converters.ToMapStructure(part.ExecutableCode)
		if err != nil {
			return nil, err
		}
		p := a2a.NewDataPart(data)
		p.Metadata = map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecutableCode}
		return p, nil
	}

	if part.FunctionCall != nil {
		data, err := converters.ToMapStructure(part.FunctionCall)
		if err != nil {
			return nil, err
		}
		p := a2a.NewDataPart(data)
		p.Metadata = map[string]any{
			a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
			a2aDataPartMetaLongRunningKey: slices.Contains(longRunningToolIDs, part.FunctionCall.ID),
		}
		return p, nil
	}

	mapStruct, err := converters.ToMapStructure(part)
	if err != nil {
		return nil, err
	}
	return a2a.NewDataPart(mapStruct), nil
}

func toGenAIContent(ctx context.Context, msg *a2a.Message, converter A2APartConverter) (*genai.Content, error) {
	if converter == nil {
		parts, err := ToGenAIParts(msg.Parts)
		if err != nil {
			return nil, err
		}
		return genai.NewContentFromParts(parts, toGenAIRole(msg.Role)), nil
	}

	parts := make([]*genai.Part, 0, len(msg.Parts))
	for _, part := range msg.Parts {
		cp, err := converter(ctx, a2a.Event(msg), part)
		if err != nil {
			return nil, err
		}
		if cp == nil {
			continue
		}
		parts = append(parts, cp)
	}
	return genai.NewContentFromParts(parts, toGenAIRole(msg.Role)), nil
}

// ToGenAIPart converts the provided A2A part to a genai equivalent.
func ToGenAIPart(part *a2a.Part) (*genai.Part, error) {
	parts, err := ToGenAIParts([]*a2a.Part{part})
	if err != nil {
		return nil, err
	}
	return parts[0], nil
}

// ToGenAIParts converts the provided A2A parts to genai equivalents.
func ToGenAIParts(parts []*a2a.Part) ([]*genai.Part, error) {
	result := make([]*genai.Part, len(parts))
	for i, part := range parts {
		if _, ok := part.Content.(a2a.Text); ok {
			r := genai.NewPartFromText(part.Text())
			if part.Metadata != nil {
				if thought, ok := part.Metadata[ToA2AMetaKey("thought")].(bool); ok {
					r.Thought = thought
				}
			}
			result[i] = r
		} else if part.Data() != nil {
			r, err := toGenAIDataPart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else if _, ok := part.Content.(a2a.Raw); ok {
			r, err := toGenAIRawPart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else if _, ok := part.Content.(a2a.URL); ok {
			r, err := toGenAIURLPart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else {
			return nil, fmt.Errorf("unknown part content type: %T", part.Content)
		}
	}
	return result, nil
}

func toGenAIRawPart(part *a2a.Part) (*genai.Part, error) {
	raw := part.Raw()
	if raw == nil {
		return nil, fmt.Errorf("raw part has no data")
	}
	data := &genai.Blob{Data: raw, MIMEType: part.MediaType, DisplayName: part.Filename}
	return &genai.Part{InlineData: data}, nil
}

func toGenAIURLPart(part *a2a.Part) (*genai.Part, error) {
	urlContent := part.URL()
	data := &genai.FileData{FileURI: string(urlContent), MIMEType: part.MediaType, DisplayName: part.Filename}
	return &genai.Part{FileData: data}, nil
}

func toGenAIDataPart(part *a2a.Part) (*genai.Part, error) {
	if part.Metadata == nil {
		return toGenAITextPart(part)
	}
	adkMetaType, ok := part.Metadata[a2aDataPartMetaTypeKey]
	if !ok {
		return toGenAITextPart(part)
	}

	bytes, err := json.Marshal(part.Data())
	if err != nil {
		return nil, err
	}

	switch adkMetaType {
	case a2aDataPartTypeCodeExecResult:
		var val genai.CodeExecutionResult
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{CodeExecutionResult: &val}, nil

	case a2aDataPartTypeFunctionCall:
		var val genai.FunctionCall
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{FunctionCall: &val}, nil

	case a2aDataPartTypeCodeExecutableCode:
		var val genai.ExecutableCode
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{ExecutableCode: &val}, nil

	case a2aDataPartTypeFunctionResponse:
		var val genai.FunctionResponse
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{FunctionResponse: &val}, nil

	default:
		return &genai.Part{Text: string(bytes)}, nil
	}
}

func toGenAITextPart(part *a2a.Part) (*genai.Part, error) {
	bytes, err := json.Marshal(part.Data())
	if err != nil {
		return nil, err
	}
	return &genai.Part{Text: string(bytes)}, nil
}
