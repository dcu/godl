package godl

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MaxPool2DOpts struct {
	Kernel  tensor.Shape
	Padding []int
	Stride  []int
}

func (opts *MaxPool2DOpts) setDefaults() {
	if opts.Padding == nil {
		opts.Padding = []int{0, 0}
	}

	if opts.Stride == nil {
		opts.Stride = []int(opts.Kernel)
	}
}

type GlobalMaxPool2DModule struct {
	model *Model
	layer LayerType
}

func (m *GlobalMaxPool2DModule) Name() string {
	return "GlobalMaxPool2d"
}

func (m *GlobalMaxPool2DModule) Forward(inputs ...*Node) Nodes {
	err := m.model.CheckArity(m.layer, inputs, 1)
	if err != nil {
		panic(err)
	}

	x := inputs[0]
	x = gorgonia.Must(gorgonia.GlobalAveragePool2D(x))

	return Nodes{x}
}

// GlobalMaxPool2D applies the global average pool operation to the given image
func GlobalMaxPool2D(nn *Model) *GlobalMaxPool2DModule {
	lt := AddLayer("GlobalMaxPool2d")

	return &GlobalMaxPool2DModule{
		model: nn,
		layer: lt,
	}
}

type MaxPool2DModule struct {
	model *Model
	opts  MaxPool2DOpts
	layer LayerType
}

func (m *MaxPool2DModule) Name() string {
	return "MaxPool2d"
}

func (m *MaxPool2DModule) Forward(inputs ...*Node) Nodes {
	err := m.model.CheckArity(m.layer, inputs, 1)
	if err != nil {
		panic(err)
	}

	x := inputs[0]
	x = gorgonia.Must(gorgonia.MaxPool2D(x, m.opts.Kernel, m.opts.Padding, m.opts.Stride))

	return Nodes{x}
}

// MaxPool2D applies the average pool operation to the given image
func MaxPool2D(nn *Model, opts MaxPool2DOpts) *MaxPool2DModule {
	lt := AddLayer("MaxPool2D")

	opts.setDefaults()

	return &MaxPool2DModule{
		model: nn,
		opts:  opts,
		layer: lt,
	}
}

var (
	_ Module = &MaxPool2DModule{}
	_ Module = &GlobalMaxPool2DModule{}
)
