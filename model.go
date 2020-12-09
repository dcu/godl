package tabnet

import "gorgonia.org/gorgonia"

const bufferSizeModel = 16

// Model implements the tab net model
type Model struct {
	g          *gorgonia.ExprGraph
	learnables []*gorgonia.Node

	model map[string]gorgonia.Value
}

// NewModel creates a new model for the neural network
func NewModel() *Model {
	return &Model{
		g:          gorgonia.NewGraph(),
		learnables: make([]*gorgonia.Node, 0, bufferSizeModel),
		model:      make(map[string]gorgonia.Value, bufferSizeModel),
	}
}
