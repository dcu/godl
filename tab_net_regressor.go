package tabnet

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type TabNetRegressor struct {
	model *Model
	layer Layer
}

type TabNetRegressorOpts struct {
	BatchSize        int
	VirtualBatchSize int
	MaskFunction     ActivationFn
	WithBias         bool

	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func NewTabNetRegressor(inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts TabNetRegressorOpts) *TabNetRegressor {
	nn := NewModel()

	embedder := nn.EmbeddingGenerator(inputDim, catDims, catIdxs, catEmbDim, EmbeddingOpts{
		WeightsInit: opts.WeightsInit,
	})

	embedDimSum := 0
	for _, v := range catEmbDim {
		embedDimSum += v
	}

	tabNetInputDim := inputDim + embedDimSum - len(catEmbDim)
	tn := nn.TabNet(TabNetOpts{
		OutputDimension:  1,
		BatchSize:        opts.BatchSize,
		VirtualBatchSize: opts.VirtualBatchSize,
		InputDimension:   tabNetInputDim,
		MaskFunction:     gorgonia.Sigmoid,
		WithBias:         opts.WithBias,
		WeightsInit:      opts.WeightsInit,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
	})

	layer := nn.Sequential(embedder, tn)

	return &TabNetRegressor{
		model: nn,
		layer: layer,
	}
}

func (r *TabNetRegressor) Train(trainX tensor.Tensor, trainY tensor.Tensor, opts TrainOpts) error {
	lambdaSparse := gorgonia.NewConstant(1e-3)

	if opts.CostFn == nil {
		opts.CostFn = func(output *gorgonia.Node, innerLoss *gorgonia.Node, y *gorgonia.Node) *gorgonia.Node {
			cost := MSELoss(output, y, MSELossOpts{})
			cost = gorgonia.Must(gorgonia.Sub(cost, gorgonia.Must(gorgonia.Mul(lambdaSparse, innerLoss))))

			return cost
		}
	}

	return r.model.Train(r.layer, trainX, trainY, opts)
}
