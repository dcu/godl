package tabnet

import "gorgonia.org/gorgonia"

// TabNet implements the tab net model
type TabNet struct {
	g          *gorgonia.ExprGraph
	learnables []*gorgonia.Node

	model map[string]gorgonia.Value
}
