package tabnet

import "gorgonia.org/gorgonia"

// Layer defines a layer on the network
type Layer func(nodes ...*gorgonia.Node) (*gorgonia.Node, error)
