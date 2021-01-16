package tabnet

import "gorgonia.org/gorgonia"

// Layer defines a layer on the network
type Layer func(inputs ...*gorgonia.Node) (output *gorgonia.Node, loss *gorgonia.Node, err error)
