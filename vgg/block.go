package vgg

import (
	"math"

	"github.com/dcu/tabnet"
	"github.com/dcu/tabnet/storage"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type BlockOpts struct {
	Channels        int
	InputDimension  int
	OutputDimension int

	ActivationFn tabnet.ActivationFn
	Dropout      float64
	Kernel       tensor.Shape
	Pad          []int
	Stride       []int
	Dilation     []int
	WithBias     bool
	WithPooling  bool

	WeightsInit, BiasInit gorgonia.InitWFn

	weightsName, biasName string
	loader                storage.Storage
}

func (o *BlockOpts) setDefaults() {
	if o.ActivationFn == nil {
		o.ActivationFn = gorgonia.Rectify
	}

	if o.Kernel == nil {
		o.Kernel = tensor.Shape{3, 3}
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
		k := math.Sqrt(1 / float64(o.OutputDimension*o.Kernel[0]*o.Kernel[1]))
		o.WeightsInit = gorgonia.Uniform(-k, k)
	}

	if o.BiasInit == nil {
		k := math.Sqrt(1 / float64(o.OutputDimension*o.Kernel[0]*o.Kernel[1]))
		o.WeightsInit = gorgonia.Uniform(-k, k)
	}
}

// VGG block composed of conv2d+maxpool with optional dropout and activation function
func Block(m *tabnet.Model, opts BlockOpts) tabnet.Layer {
	opts.setDefaults()

	lt := tabnet.AddLayer("vgg.Block")

	w := m.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension, opts.Channels, opts.Channels}, tabnet.NewNodeOpts{
		InitFN: opts.WeightsInit,
	})

	var bias *gorgonia.Node
	if opts.WithBias {
		bias = m.AddWeights(lt, tensor.Shape{1, opts.OutputDimension, 1, 1}, tabnet.NewNodeOpts{
			InitFN: opts.BiasInit,
		})
	}

	return func(inputs ...*gorgonia.Node) (tabnet.Result, error) {
		if err := m.CheckArity(lt, inputs, 1); err != nil {
			return tabnet.Result{}, err
		}

		x := inputs[0]
		x = gorgonia.Must(gorgonia.Conv2d(x, w, opts.Kernel, opts.Pad, opts.Stride, opts.Dilation))

		if bias != nil {
			x = gorgonia.Must(gorgonia.BroadcastAdd(x, bias, nil, []byte{0, 2, 3}))
		}

		if opts.ActivationFn != nil {
			x = gorgonia.Must(opts.ActivationFn(x))
		}

		if opts.WithPooling {
			x = gorgonia.Must(gorgonia.MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}))
		}

		if opts.Dropout > 0.0 {
			x = gorgonia.Must(gorgonia.Dropout(x, opts.Dropout))
		}

		return tabnet.Result{
			Output: x,
		}, nil
	}
}
