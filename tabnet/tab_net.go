package tabnet

import (
	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

type TabNetOpts struct {
	OutputSize int
	InputSize  int
	BatchSize  int

	SharedBlocks       int
	IndependentBlocks  int
	DecisionSteps      int
	PredictionLayerDim int
	AttentionLayerDim  int

	MaskFunction godl.ActivationFn

	WithBias bool

	Gamma                            float32
	Momentum                         float32
	Epsilon                          float32
	VirtualBatchSize                 int
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn

	CatDims   []int
	CatIdxs   []int
	CatEmbDim []int
}

func TabNet(nn *godl.Model, opts TabNetOpts) godl.Layer {
	embedder := godl.EmbeddingGenerator(nn, opts.InputSize, opts.CatDims, opts.CatIdxs, opts.CatEmbDim, godl.EmbeddingOpts{
		WeightsInit: opts.WeightsInit,
	})

	embedDimSum := 0
	for _, v := range opts.CatEmbDim {
		embedDimSum += v
	}

	tabNetInputDim := opts.InputSize + embedDimSum - len(opts.CatEmbDim)
	tn := TabNetNoEmbeddings(nn, TabNetNoEmbeddingsOpts{
		InputSize:          tabNetInputDim,
		OutputSize:         opts.OutputSize,
		BatchSize:          opts.BatchSize,
		VirtualBatchSize:   opts.VirtualBatchSize,
		MaskFunction:       opts.MaskFunction,
		WithBias:           opts.WithBias,
		WeightsInit:        opts.WeightsInit,
		ScaleInit:          opts.ScaleInit,
		BiasInit:           opts.BiasInit,
		SharedBlocks:       opts.SharedBlocks,
		IndependentBlocks:  opts.IndependentBlocks,
		DecisionSteps:      opts.DecisionSteps,
		PredictionLayerDim: opts.PredictionLayerDim,
		AttentionLayerDim:  opts.AttentionLayerDim,
		Gamma:              opts.Gamma,
		Momentum:           opts.Momentum,
		Epsilon:            opts.Epsilon,
	})

	return func(nodes ...*gorgonia.Node) (godl.Result, error) {
		x := nodes[0]
		res, err := embedder(x)
		if err != nil {
			return godl.Result{}, err
		}

		return tn(res.Output)
	}
}
