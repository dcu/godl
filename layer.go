package tabnet

import "gorgonia.org/gorgonia"

// Layer defines a layer on the network
type Layer func(nodes ...*gorgonia.Node) (*gorgonia.Node, error)

// ActivationFn represents an activation function
type ActivationFn func(*gorgonia.Node) (*gorgonia.Node, error)
