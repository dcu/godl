package vgg

import (
	"log"

	"github.com/dcu/deepzen"
	"github.com/dcu/deepzen/storage/nn1"
	"gorgonia.org/gorgonia"
)

type Opts struct {
	WithBias              bool
	WeightsInit, BiasInit gorgonia.InitWFn
	Learnable             bool

	PreTrained            bool
	OnlyFeatureExtraction bool
}

func VGG16(m *deepzen.Model, opts Opts) deepzen.Layer {
	lt := deepzen.AddLayer("vgg.VGG16")

	var (
		loader *nn1.NN1
		err    error
	)

	if opts.PreTrained {
		fileName := "vgg16.nn1"
		if opts.OnlyFeatureExtraction {
			fileName = "vgg16_notop.nn1"
		}

		loader, err = nn1.Open(fileName)
		if err != nil {
			log.Panicf("couldn't load %v in pre-trained mode: %v", fileName, err)
		}
	}

	layers := []deepzen.Layer{
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  3,
			OutputDimension: 64,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  64,
			OutputDimension: 64,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  64,
			OutputDimension: 128,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  128,
			OutputDimension: 128,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  128,
			OutputDimension: 256,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  256,
			OutputDimension: 256,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  256,
			OutputDimension: 256,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  256,
			OutputDimension: 512,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  512,
			OutputDimension: 512,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  512,
			OutputDimension: 512,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  512,
			OutputDimension: 512,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  512,
			OutputDimension: 512,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  512,
			OutputDimension: 512,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
			WithPooling:     true,
			loader:          loader,
			weightsName:     "/block1_conv1/block1_conv1_W_1:0",
			biasName:        "/block1_conv1/block1_conv1_b_1:0",
		}),
	}

	if !opts.OnlyFeatureExtraction {
		layers = append(layers,
			deepzen.FC(m, deepzen.FCOpts{
				InputDimension:  25088,
				OutputDimension: 4096,
				WithBias:        opts.WithBias,
				Activation:      gorgonia.Rectify,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
			}),
			deepzen.FC(m, deepzen.FCOpts{
				InputDimension:  4096,
				OutputDimension: 4096,
				WithBias:        opts.WithBias,
				Activation:      gorgonia.Rectify,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
			}),
			deepzen.FC(m, deepzen.FCOpts{
				InputDimension:  4096,
				OutputDimension: 1000,
				WithBias:        opts.WithBias,
				Activation:      gorgonia.Rectify,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
			}),
		)
	}

	seq := deepzen.Sequential(m, layers...)

	return func(inputs ...*gorgonia.Node) (deepzen.Result, error) {
		if err := m.CheckArity(lt, inputs, 1); err != nil {
			return deepzen.Result{}, err
		}

		x := inputs[0]

		return seq(x)
	}
}
