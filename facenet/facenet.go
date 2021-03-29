package facenet

import (
	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Opts struct {
	WithBias              bool
	WeightsInit, BiasInit gorgonia.InitWFn
	Learnable             bool

	PreTrained            bool
	OnlyFeatureExtraction bool
	Classes               int
}

func (opts *Opts) setDefaults() {
	if opts.Classes == 0 {
		opts.Classes = 8631
	}
}

func FaceNet(opts Opts) func(m *godl.Model) godl.Layer {
	return func(m *godl.Model) godl.Layer {
		return FaceNetLayer(m, opts)
	}
}

func FaceNetLayer(m *godl.Model, opts Opts) godl.Layer {
	lt := godl.AddLayer("FaceNet")

	blocks := []godl.Layer{
		// Stage 2
		ConvBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{64, 64, 256},
			Stage:      2,
			Block:      1,
			Stride:     []int{1, 1},
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{64, 64, 256},
			Stage:      2,
			Block:      2,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{64, 64, 256},
			Stage:      2,
			Block:      3,
		}),
		// Stage 3
		ConvBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{128, 128, 512},
			Stage:      3,
			Block:      1,
			Stride:     []int{1, 1},
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{128, 128, 512},
			Stage:      3,
			Block:      2,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{128, 128, 512},
			Stage:      3,
			Block:      3,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{128, 128, 512},
			Stage:      3,
			Block:      4,
		}),
		// Stage 4
		ConvBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{256, 256, 1024},
			Stage:      4,
			Block:      1,
			Stride:     []int{1, 1},
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{256, 256, 1024},
			Stage:      4,
			Block:      2,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{256, 256, 1024},
			Stage:      4,
			Block:      3,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{256, 256, 1024},
			Stage:      4,
			Block:      4,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{256, 256, 1024},
			Stage:      4,
			Block:      5,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{256, 256, 1024},
			Stage:      4,
			Block:      6,
		}),
		// Stage 5
		ConvBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{512, 512, 2048},
			Stage:      5,
			Block:      1,
			Stride:     []int{1, 1},
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{512, 512, 2048},
			Stage:      5,
			Block:      2,
		}),
		IdentityBlock(m, BlockOpts{
			KernelSize: tensor.Shape{3, 3},
			Filters:    [3]int{512, 512, 2048},
			Stage:      5,
			Block:      3,
		}),
		godl.AvgPool2D(m, godl.AvgPool2DOpts{
			Kernel: tensor.Shape{7, 7},
		}),
	}

	if !opts.OnlyFeatureExtraction {
		blocks = append(blocks, godl.FC(m, godl.FCOpts{
			InputDimension:  0, // FIXME
			OutputDimension: opts.Classes,
		}))
	} else {
		// TODO: give option to apply global max pool2d
		blocks = append(blocks, godl.GlobalAvgPool2D(m))
	}

	seq := godl.Sequential(m, blocks...)

	return func(inputs ...*gorgonia.Node) (godl.Result, error) {
		if err := m.CheckArity(lt, inputs, 1); err != nil {
			return godl.Result{}, err
		}

		x := inputs[0]

		result, err := godl.Conv2d(m, godl.Conv2dOpts{
			InputDimension:  64,
			OutputDimension: 3,
			KernelSize:      tensor.Shape{7, 7},
			Pad:             []int{0, 0},
		})(x)
		handleErr(err)

		result, err = godl.BatchNorm2d(m, godl.BatchNormOpts{
			InputSize: result.Output.Shape()[0],
		})(result.Output)

		x = gorgonia.Must(gorgonia.Rectify(result.Output))
		x = gorgonia.Must(gorgonia.MaxPool2D(x, tensor.Shape{3, 3}, []int{0, 0}, []int{1, 1}))

		result, err = seq(x)
		handleErr(err)

		return godl.Result{}, nil
	}
}

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}
