package vggface2

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

type VGGFace2Module struct {
	model *godl.Model
	layer godl.LayerType

	seq godl.ModuleList
}

func (m *VGGFace2Module) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]

	result := godl.Conv2d(m.model, godl.Conv2dOpts{
		InputDimension:  64,
		OutputDimension: 3,
		KernelSize:      tensor.Shape{7, 7},
		Pad:             []int{0, 0},
		WeightsName:     "/conv1/7x7_s2/gamma",
		BiasName:        "/conv1/7x7_s2/beta",
	}).Forward(x)

	result = godl.BatchNorm2d(m.model, godl.BatchNormOpts{
		InputSize: result[0].Shape()[0],
		ScaleName: "/conv1/7x7_s2/bn/gamma",
		BiasName:  "/conv1/7x7_s2/bn/beta",
	}).Forward(result[0])

	x = gorgonia.Must(gorgonia.Rectify(result[0]))
	x = gorgonia.Must(gorgonia.MaxPool2D(x, tensor.Shape{3, 3}, []int{0, 0}, []int{1, 1}))

	result = m.seq.Forward(x)

	return result
}

func VGGFace2Builder(opts Opts) func(*godl.Model) godl.Module {
	return func(m *godl.Model) godl.Module {
		return VGGFace2(m, opts)
	}
}

func VGGFace2(m *godl.Model, opts Opts) *VGGFace2Module {
	lt := godl.AddLayer("VGGFace2")

	blocks := []godl.Module{
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
		blocks = append(blocks, godl.Linear(m, godl.LinearOpts{
			InputDimension:  0, // FIXME
			OutputDimension: opts.Classes,
			WeightsName:     "classifier/kernel",
			BiasName:        "classifier/bias",
		}))
	} else {
		// TODO: give option to apply global max pool2d
		blocks = append(blocks, godl.GlobalAvgPool2D(m))
	}

	seq := godl.Sequential(m, blocks...)

	return &VGGFace2Module{
		model: m,
		layer: lt,
		seq:   seq,
	}
}

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}
