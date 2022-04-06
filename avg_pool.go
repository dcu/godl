package godl

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type AvgPool2DOpts struct {
	Kernel  tensor.Shape
	Padding []int
	Stride  []int
}

func (opts *AvgPool2DOpts) setDefaults() {
	if opts.Padding == nil {
		opts.Padding = []int{0, 0}
	}

	if opts.Stride == nil {
		opts.Padding = []int(opts.Kernel)
	}
}

type GlobalAvgPool2DModule struct {
	model *Model
	layer LayerType
}

func (m *GlobalAvgPool2DModule) Forward(inputs ...*Node) Nodes {
	err := m.model.CheckArity(m.layer, inputs, 1)
	if err != nil {
		panic(err)
	}

	x := inputs[0]
	x = gorgonia.Must(gorgonia.GlobalAveragePool2D(x))

	return Nodes{x}
}

// GlobalAvgPool2D applies the global average pool operation to the given image
func GlobalAvgPool2D(nn *Model) *GlobalAvgPool2DModule {
	lt := AddLayer("GlobalAvgPool2D")

	return &GlobalAvgPool2DModule{
		model: nn,
		layer: lt,
	}
}

type AvgPool2DModule struct {
	model *Model
	opts  AvgPool2DOpts
	layer LayerType
}

func (m *AvgPool2DModule) Forward(inputs ...*Node) Nodes {
	err := m.model.CheckArity(m.layer, inputs, 1)
	if err != nil {
		panic(err)
	}

	x := inputs[0]
	x = gorgonia.Must(gorgonia.AveragePool2D(x, m.opts.Kernel, m.opts.Padding, m.opts.Stride))

	return Nodes{x}
}

// AvgPool2D applies the average pool operation to the given image
func AvgPool2D(nn *Model, opts AvgPool2DOpts) *AvgPool2DModule {
	lt := AddLayer("AvgPool2D")

	return &AvgPool2DModule{
		model: nn,
		opts:  opts,
		layer: lt,
	}
}
