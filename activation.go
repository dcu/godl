package tabnet

import "gorgonia.org/gorgonia"

// ActivationFn represents an activation function
type ActivationFn func(*gorgonia.Node) (*gorgonia.Node, error)

// this is wrap because gorgonia.Sparsemax doesn't match the ActivationFn signature
func sparsemax(x *gorgonia.Node) (*gorgonia.Node, error) {
	return gorgonia.Sparsemax(x)
}

var (
	_ ActivationFn = sparsemax
)
