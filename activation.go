package godl

import (
	"gorgonia.org/gorgonia"
)

type ActivationModule struct {
	name string
	fn   func(x *Node) (*Node, error)
}

func (m *ActivationModule) Name() string {
	return m.name
}

func (m *ActivationModule) Forward(inputs ...*Node) Nodes {
	x := inputs[0]
	y := gorgonia.Must(m.fn(x))

	return Nodes{y}
}

type ActivationAxisModule struct {
	name string
	axis []int
	fn   func(x *Node, axis ...int) (*Node, error)
}

func (m *ActivationAxisModule) Forward(inputs ...*Node) Nodes {
	x := inputs[0]
	y := gorgonia.Must(m.fn(x, m.axis...))

	return Nodes{y}
}

func (m *ActivationAxisModule) Name() string {
	return m.name
}

func Sigmoid() Module {
	return &ActivationModule{
		name: "Sigmoid",
		fn:   gorgonia.Sigmoid,
	}
}

func Tanh() Module {
	return &ActivationModule{
		name: "Tanh",
		fn:   gorgonia.Tanh,
	}
}

func Rectify() Module {
	return &ActivationModule{
		name: "Rectify",
		fn:   gorgonia.Rectify,
	}
}

func SparseMax(axis ...int) Module {
	return &ActivationAxisModule{
		name: "SparseMax",
		axis: axis,
		fn:   gorgonia.Sparsemax,
	}
}

func SoftMax(axis ...int) Module {
	return &ActivationAxisModule{
		name: "SoftMax",
		axis: axis,
		fn:   gorgonia.SoftMax,
	}
}
