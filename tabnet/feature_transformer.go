package tabnet

import (
	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

// FeatureTransformerOpts contains options for feature transformer layer
type FeatureTransformerOpts struct {
	Shared            []*godl.LinearModule
	VirtualBatchSize  int
	IndependentBlocks int
	InputDimension    int
	OutputDimension   int
	WithBias          bool
	Momentum          float64

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

type FeatureTransformerModule struct {
	model       *godl.Model
	layer       godl.LayerType
	opts        FeatureTransformerOpts
	shared      *GLUBlockModule
	independent *GLUBlockModule
}

func (m *FeatureTransformerModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]
	res := m.shared.Forward(x)

	return m.independent.Forward(res[0])
}

// FeatureTransformer implements a feature transformer layer
func FeatureTransformer(nn *godl.Model, opts FeatureTransformerOpts) *FeatureTransformerModule {
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

	return &FeatureTransformerModule{
		model:       nn,
		layer:       lt,
		opts:        opts,
		shared:      shared,
		independent: independent,
	}
}
