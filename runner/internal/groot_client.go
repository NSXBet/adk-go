package internal

import (
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"
)

type Client struct {
	conn *websocket.Conn
}

type Port struct {
	Name     string `json:"name,omitempty"`
	StreamID string `json:"streamId,omitempty"`
}

type ActionGraph struct {
	Actions []*Action `json:"actions,omitempty"`
	Outputs []*Port   `json:"outputs,omitempty"`
}

type Action struct {
	Name    string  `json:"name,omitempty"`
	Inputs  []*Port `json:"inputs,omitempty"`
	Outputs []*Port `json:"outputs,omitempty"`
	// TODO: Add configs.
}

type Chunk struct {
	MIMEType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`
	// TODO: Add metadata.
}

type StreamFrame struct {
	StreamID  string `json:"streamId,omitempty"`
	Data      *Chunk `json:"data,omitempty"`
	Continued bool   `json:"continued,omitempty"`
}

type executeActionsMsg struct {
	SessionID    string         `json:"sessionId,omitempty"`
	ActionGraph  *ActionGraph   `json:"actionGraph,omitempty"`
	StreamFrames []*StreamFrame `json:"streamFrames,omitempty"`
}

type ShadowStatus string

const (
	ShadowStatusUnknown   = "unknown"
	ShadowStatusPending   = "pending"
	ShadowStatusCompleted = "completed"
)

type Shadow struct {
	sess         *Session
	displayName  string
	input        string
	output       string
	waitStreamID string
}

func (s *Shadow) WriteFrame(id string, c *Chunk, continued bool) error {
	return s.sess.writeStreamFrame(&StreamFrame{
		StreamID:  id,
		Data:      c,
		Continued: continued,
	})
}

func (s *Shadow) Wait() error {
	var resp executeActionsMsg
	for {
		if err := s.sess.c.conn.ReadJSON(&resp); err != nil {
			return err
		}
		for _, frame := range resp.StreamFrames {
			if frame.StreamID == s.waitStreamID && !frame.Continued {
				return nil
			}
		}
	}
}

func NewClient(endpoint string, apiKey string) (*Client, error) {
	c, _, err := websocket.DefaultDialer.Dial(endpoint+"?key="+apiKey, nil)
	if err != nil {
		return nil, err
	}
	return &Client{conn: c}, nil
}

type Session struct {
	c         *Client
	sessionID string
}

func (c *Client) OpenSession(sessionID string) (*Session, error) {
	return &Session{c: c, sessionID: sessionID}, nil
}

func (s *Session) NewADKShadow(name string, input string, output string) (*Shadow, ShadowStatus, error) {
	waitID, err := s.writeShadowAction(name, input, output)
	if err != nil {
		return nil, ShadowStatusUnknown, err
	}
	return &Shadow{
		sess:         s,
		displayName:  name,
		input:        input,
		output:       output,
		waitStreamID: waitID,
	}, ShadowStatusPending, nil
}

func (s *Session) writeStreamFrame(sf *StreamFrame) error {
	return s.c.conn.WriteJSON(&executeActionsMsg{
		StreamFrames: []*StreamFrame{sf},
	})
}

func (s *Session) writeShadowAction(name string, input, output string) (waitID string, err error) {
	waitID = uuid.NewString()
	if err := s.c.conn.WriteJSON(&executeActionsMsg{
		SessionID: s.sessionID,
		ActionGraph: &ActionGraph{
			Actions: []*Action{
				{
					Name:    "save_stream",
					Inputs:  []*Port{{Name: "input", StreamID: output}},
					Outputs: []*Port{{Name: "output", StreamID: waitID}},
				},
			},
			Outputs: []*Port{{Name: "output", StreamID: waitID}},
		},
	}); err != nil {
		return "", err
	}
	return waitID, err
}

func ChunkFromPart(p *genai.Part) *Chunk {
	var chunk Chunk
	switch {
	case p.Text != "":
		chunk.MIMEType = "text/plain"
		chunk.Data = []byte(p.Text)
	case p.InlineData != nil:
		chunk.MIMEType = p.InlineData.MIMEType
		chunk.Data = p.InlineData.Data
	default:
		// TODO(jbd): Support all types.
		panic("part not supported yet")
	}
	return &chunk
}
