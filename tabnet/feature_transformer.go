package tabnet

import (
	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

// FeatureTransformerOpts contains options for feature transformer layer
type FeatureTransformerOpts struct {
	Shared            []godl.Layer
	VirtualBatchSize  int
	IndependentBlocks int
	InputDimension    int
	OutputDimension   int
	WithBias          bool
	Momentum          float32

	WeightsInit gorgonia.InitWFn
}

func (o *FeatureTransformerOpts) setDefaults() {
	if o.InputDimension == 0 {
		panic("input dimension can't be nil")
	}

	if o.OutputDimension == 0 {
		panic("output dimension can't be nil")
	}

	if o.Momentum == 0 {
		o.Momentum = 0.01
	}
}

// FeatureTransformer implements a feature transformer layer
func FeatureTransformer(nn *godl.Model, opts FeatureTransformerOpts) godl.Layer {
	lt := godl.AddLayer("FeatureTransformer")

	opts.setDefaults()

	shared := GLUBlock(nn, GLUBlockOpts{
		InputDimension:   opts.InputDimension,
		OutputDimension:  opts.OutputDimension,
		VirtualBatchSize: opts.VirtualBatchSize,
		Size:             len(opts.Shared),
		Shared:           opts.Shared,
		WithBias:         opts.WithBias,
		Momentum:         opts.Momentum,
		WeightsInit:      opts.WeightsInit,
	})

	independent := GLUBlock(nn, GLUBlockOpts{
		InputDimension:   opts.InputDimension,
		OutputDimension:  opts.OutputDimension,
		VirtualBatchSize: opts.VirtualBatchSize,
		Size:             opts.IndependentBlocks,
		Shared:           nil,
		WithBias:         opts.WithBias,
		Momentum:         opts.Momentum,
		WeightsInit:      opts.WeightsInit,
	})

	return func(nodes ...*gorgonia.Node) (godl.Result, error) {
		if err := nn.CheckArity(lt, nodes, 1); err != nil {
			return godl.Result{}, err
		}

		x := nodes[0]

		res, err := shared(x)
		if err != nil {
			return godl.Result{}, err
		}

		return independent(res.Output)
	}
}
