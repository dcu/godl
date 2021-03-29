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

// AvgPool2D applies the average pool operation to the given image
func AvgPool2D(nn *Model, opts AvgPool2DOpts) Layer {
	lt := AddLayer("AvgPool2D")

	return func(inputs ...*gorgonia.Node) (Result, error) {
		err := nn.CheckArity(lt, inputs, 1)
		if err != nil {
			return Result{}, err
		}

		x := inputs[0]
		x = gorgonia.Must(gorgonia.AveragePool2D(x, opts.Kernel, opts.Padding, opts.Stride))

		return Result{Output: x}, nil
	}
}

// GlobalAvgPool2D applies the global average pool operation to the given image
func GlobalAvgPool2D(nn *Model) Layer {
	lt := AddLayer("GlobalAvgPool2D")

	return func(inputs ...*gorgonia.Node) (Result, error) {
		err := nn.CheckArity(lt, inputs, 1)
		if err != nil {
			return Result{}, err
		}

		x := inputs[0]
		x = gorgonia.Must(gorgonia.GlobalAveragePool2D(x))

		return Result{Output: x}, nil
	}
}
