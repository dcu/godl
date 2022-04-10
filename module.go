package godl

import (
	"gorgonia.org/gorgonia"
)

type (
	Node  = gorgonia.Node
	Nodes = gorgonia.Nodes
)

type Module interface {
	Forward(inputs ...*Node) Nodes
	Name() string
}

type ModuleList []Module

func (m *ModuleList) Add(mods ...Module) {
	*m = append(*m, mods...)
}

func (m ModuleList) Forward(inputs ...*Node) (out Nodes) {
	out = inputs

	for _, mod := range m {
		out = mod.Forward(out...)
	}

	return out
}

func (m ModuleList) Name() string {
	return "ModuleList"
}

var (
	_ Module = ModuleList{}
)
