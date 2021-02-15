package vgg

import (
	"github.com/dcu/tabnet"
	"gorgonia.org/gorgonia"
)

type Opts struct {
	WithBias              bool
	WeightsInit, BiasInit gorgonia.InitWFn
	Learnable             bool

	OnlyFeatureExtraction bool
}

func VGG16(m *tabnet.Model, opts Opts) tabnet.Layer {
	lt := tabnet.AddLayer("vgg.VGG16")

	layers := []tabnet.Layer{
		Block(m, BlockOpts{
			Channels:        3,
			InputDimension:  3,
			OutputDimension: 64,
			ActivationFn:    gorgonia.Rectify,
			Dropout:         0.0,
			WithBias:        opts.WithBias,
			BiasInit:        opts.BiasInit,
			WeightsInit:     opts.WeightsInit,
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
		}),
	}

	if !opts.OnlyFeatureExtraction {
		layers = append(layers,
			tabnet.FC(m, tabnet.FCOpts{
				InputDimension:  25088,
				OutputDimension: 4096,
				WithBias:        opts.WithBias,
				Activation:      gorgonia.Rectify,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
			}),
			tabnet.FC(m, tabnet.FCOpts{
				InputDimension:  4096,
				OutputDimension: 4096,
				WithBias:        opts.WithBias,
				Activation:      gorgonia.Rectify,
				Dropout:         0.0,
				WeightsInit:     opts.WeightsInit,
				BiasInit:        opts.BiasInit,
			}),
			tabnet.FC(m, tabnet.FCOpts{
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

	seq := tabnet.Sequential(m, layers...)

	return func(inputs ...*gorgonia.Node) (tabnet.Result, error) {
		if err := m.CheckArity(lt, inputs, 1); err != nil {
			return tabnet.Result{}, err
		}

		x := inputs[0]

		return seq(x)
	}
}
