package internal

import "google.golang.org/adk/agent"

type Registry struct {
	agents     map[string]agent.Agent
	agentNames map[agent.Agent]string

	// TODO: Register tools
}

func NewRegistry(rootAgent agent.Agent) *Registry {
	r := &Registry{
		agents:     make(map[string]agent.Agent),
		agentNames: make(map[agent.Agent]string),
	}
	r.register("", rootAgent)
	return r
}

func (r *Registry) register(prefix string, agent agent.Agent) {
	name := agent.Name()
	fullname := prefix + name
	r.agents[fullname] = agent
	r.agentNames[agent] = fullname
	for _, subAgent := range agent.SubAgents() {
		r.register(fullname+".", subAgent)
	}
}

func (r *Registry) AgentFullname(agent agent.Agent) string {
	return r.agentNames[agent]
}

func (r *Registry) Resolve(fullname string) agent.Agent {
	return r.agents[fullname]
}
