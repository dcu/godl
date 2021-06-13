package vgg

import (
	"math"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// BlockOpts are the options for a VGG Block
type BlockOpts struct {
	InputDimension  int
	OutputDimension int

	ActivationFn godl.ActivationFn
	Dropout      float64
	KernelSize   tensor.Shape
	Pad          []int
	Stride       []int
	Dilation     []int
	WithBias     bool
	WithPooling  bool

	WeightsInit, BiasInit gorgonia.InitWFn
	WeightsName, BiasName string
	FixedWeights          bool
}

func (o *BlockOpts) setDefaults() {
	if o.ActivationFn == nil {
		o.ActivationFn = gorgonia.Rectify
	}

	if o.KernelSize == nil {
		o.KernelSize = tensor.Shape{3, 3}
	}

	if o.Pad == nil {
		o.Pad = []int{1, 1}
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

// Block is a VGG block composed of conv2d+maxpool with optional dropout and activation function
func Block(m *godl.Model, opts BlockOpts) godl.Layer {
	opts.setDefaults()

	lt := godl.AddLayer("vgg.Block")

	w := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, opts.KernelSize[0], opts.KernelSize[0]}, godl.NewWeightsOpts{
		InitFN:     opts.WeightsInit,
		UniqueName: opts.WeightsName,
		Fixed:      opts.FixedWeights,
	})

	var bias *gorgonia.Node
	if opts.WithBias {
		bias = m.AddBias(lt, tensor.Shape{1, opts.OutputDimension, 1, 1}, godl.NewWeightsOpts{
			InitFN:     opts.BiasInit,
			UniqueName: opts.BiasName,
			Fixed:      opts.FixedWeights,
		})
	}

	return func(inputs ...*gorgonia.Node) (godl.Result, error) {
		if err := m.CheckArity(lt, inputs, 1); err != nil {
			return godl.Result{}, err
		}

		x := inputs[0]
		x = gorgonia.Must(gorgonia.Conv2d(x, w, opts.KernelSize, opts.Pad, opts.Stride, opts.Dilation))

		if bias != nil {
			x = gorgonia.Must(gorgonia.BroadcastAdd(x, bias, nil, []byte{0, 2, 3}))
		}

		if opts.ActivationFn != nil {
			x = gorgonia.Must(opts.ActivationFn(x))
		}

		if opts.WithPooling {
			x = gorgonia.Must(gorgonia.MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}))
		}

		if opts.Dropout > 0.0 && m.Training {
			x = gorgonia.Must(gorgonia.Dropout(x, opts.Dropout))
		}

		return godl.Result{
			Output: x,
		}, nil
	}
}
