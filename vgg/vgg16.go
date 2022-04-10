package vgg

import (
	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
)

// Opts are the options for VGG
type Opts struct {
	WithBias              bool
	WeightsInit, BiasInit gorgonia.InitWFn
	Learnable             bool

	PreTrained            bool
	OnlyFeatureExtraction bool
}

type VGG16Module struct {
	model *godl.Model
	opts  Opts
	layer godl.LayerType

	seq godl.ModuleList
}

func (m *VGG16Module) Name() string {
	return "VGG16"
}

func (m *VGG16Module) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]

	return m.seq.Forward(x)
}

func VGG16Builder(opts Opts) func(*godl.Model) godl.Module {
	return func(m *godl.Model) godl.Module {
		return VGG16(m, opts)
	}
}

// VGG16 returns the layer for the VGG16 network
func VGG16(m *godl.Model, opts Opts) *VGG16Module {
	lt := godl.AddLayer("vgg.VGG16")
	fixedWeights := false

	if opts.PreTrained {
		fileName := "vgg16.nn1"
		if opts.OnlyFeatureExtraction {
			fileName = "vgg16_weights_th_dim_ordering_th_kernels_notop.nn1"
		}

		err := m.Storage.LoadFile(fileName)
		if err != nil {
			panic(err)
		}

		fixedWeights = !opts.Learnable
		opts.WithBias = true
	}

	layers := []godl.Module{
		Block(m, BlockOpts{
			InputDimension:  3,
			OutputDimension: 64,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block1_conv1/block1_conv1_W:0",
			BiasName:        "/block1_conv1/block1_conv1_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  64,
			OutputDimension: 64,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			WeightsName:     "/block1_conv2/block1_conv2_W:0",
			BiasName:        "/block1_conv2/block1_conv2_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  64,
			OutputDimension: 128,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block2_conv1/block2_conv1_W:0",
			BiasName:        "/block2_conv1/block2_conv1_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  128,
			OutputDimension: 128,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			WeightsName:     "/block2_conv2/block2_conv2_W:0",
			BiasName:        "/block2_conv2/block2_conv2_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  128,
			OutputDimension: 256,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block3_conv1/block3_conv1_W:0",
			BiasName:        "/block3_conv1/block3_conv1_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  256,
			OutputDimension: 256,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block3_conv2/block3_conv2_W:0",
			BiasName:        "/block3_conv2/block3_conv2_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  256,
			OutputDimension: 256,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			WeightsName:     "/block3_conv3/block3_conv3_W:0",
			BiasName:        "/block3_conv3/block3_conv3_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  256,
			OutputDimension: 512,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block4_conv1/block4_conv1_W:0",
			BiasName:        "/block4_conv1/block4_conv1_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  512,
			OutputDimension: 512,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block4_conv2/block4_conv2_W:0",
			BiasName:        "/block4_conv2/block4_conv2_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  512,
			OutputDimension: 512,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			WeightsName:     "/block4_conv3/block4_conv3_W:0",
			BiasName:        "/block4_conv3/block4_conv3_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  512,
			OutputDimension: 512,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block5_conv1/block5_conv1_W:0",
			BiasName:        "/block5_conv1/block5_conv1_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  512,
			OutputDimension: 512,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WeightsName:     "/block5_conv2/block5_conv2_W:0",
			BiasName:        "/block5_conv2/block5_conv2_b:0",
			FixedWeights:    fixedWeights,
		}),
		Block(m, BlockOpts{
			InputDimension:  512,
			OutputDimension: 512,
			Activation:      gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			WeightsName:     "/block5_conv3/block5_conv3_W:0",
			BiasName:        "/block5_conv3/block5_conv3_b:0",
			FixedWeights:    fixedWeights,
		}),
	}

	if !opts.OnlyFeatureExtraction {
		layers = append(layers,
			godl.Linear(m, godl.LinearOpts{
				InputDimension:  25088,
				OutputDimension: 4096,
				WithBias:        opts.WithBias,
				Activation:      gorgonia.Rectify,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
				WeightsName:     "/fc1/fc1_W:0",
				BiasName:        "/fc1/fc1_b:0",
				FixedWeights:    fixedWeights,
			}),
			godl.Linear(m, godl.LinearOpts{
				InputDimension:  4096,
				OutputDimension: 4096,
				WithBias:        opts.WithBias,
				Activation:      gorgonia.Rectify,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
				WeightsName:     "/fc2/fc2_W:0",
				BiasName:        "/fc2/fc2_b:0",
				FixedWeights:    fixedWeights,
			}),
			godl.Linear(m, godl.LinearOpts{
				InputDimension:  4096,
				OutputDimension: 1000,
				WithBias:        opts.WithBias,
				Activation:      activation.SoftMax,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
				WeightsName:     "/predictions/predictions_W:0",
				BiasName:        "/predictions/predictions_b:0",
				FixedWeights:    fixedWeights,
			}),
		)
	}

	seq := godl.Sequential(m, layers...)

	return &VGG16Module{
		model: m,
		opts:  opts,
		layer: lt,
		seq:   seq,
	}
}
