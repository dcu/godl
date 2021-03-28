package godl

import (
	"math"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Conv2dOpts are the options to run the conv2d operation
type Conv2dOpts struct {
	InputDimension  int
	OutputDimension int

	KernelSize tensor.Shape
	Pad        []int
	Stride     []int
	Dilation   []int

	WithBias bool

	WeightsInit, BiasInit gorgonia.InitWFn
	WeightsName, BiasName string
	FixedWeights          bool
}

func (o *Conv2dOpts) setDefaults() {
	if o.KernelSize == nil {
		o.KernelSize = tensor.Shape{3, 3}
	}

	if o.Pad == nil {
		o.Pad = []int{0, 0}
	}

	if o.Stride == nil {
		o.Stride = []int{1, 1}
	}

	if o.Dilation == nil {
		o.Dilation = []int{1, 1}
	}

	if o.WeightsInit == nil {
		k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize[0]*o.KernelSize[1]))
		o.WeightsInit = gorgonia.Uniform(-k, k)
	}

	if o.BiasInit == nil {
		k := math.Sqrt(1 / float64(o.OutputDimension*o.KernelSize[0]*o.KernelSize[1]))
		o.WeightsInit = gorgonia.Uniform(-k, k)
	}
}

// Conv2d applies a conv2d operation to the input
func Conv2d(m *Model, opts Conv2dOpts) Layer {
	opts.setDefaults()
	lt := AddLayer("Conv2d")

	w := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, opts.KernelSize[0], opts.KernelSize[0]}, NewWeightsOpts{
		InitFN:     opts.WeightsInit,
		UniqueName: opts.WeightsName,
		Fixed:      opts.FixedWeights,
	})

	var bias *gorgonia.Node
	if opts.WithBias {
		bias = m.AddBias(lt, tensor.Shape{1, opts.OutputDimension, 1, 1}, NewWeightsOpts{
			InitFN:     opts.BiasInit,
			UniqueName: opts.BiasName,
			Fixed:      opts.FixedWeights,
		})
	}

	return func(inputs ...*gorgonia.Node) (Result, error) {
		err := m.CheckArity(lt, inputs, 1)
		if err != nil {
			return Result{}, err
		}

		x := inputs[0]
		x = gorgonia.Must(gorgonia.Conv2d(x, w, opts.KernelSize, opts.Pad, opts.Stride, opts.Dilation))

		if bias != nil {
			x = gorgonia.Must(gorgonia.BroadcastAdd(x, bias, nil, []byte{0, 2, 3}))
		}

		return Result{Output: x}, nil
	}
}
