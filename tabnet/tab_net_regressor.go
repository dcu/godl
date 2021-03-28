package tabnet

import (
	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Regressor struct {
	model *godl.Model
	layer godl.Layer
}

type RegressorOpts struct {
	BatchSize        int
	VirtualBatchSize int
	MaskFunction     godl.ActivationFn
	WithBias         bool

	SharedBlocks       int
	IndependentBlocks  int
	DecisionSteps      int
	PredictionLayerDim int
	AttentionLayerDim  int

	Gamma    float32
	Momentum float32
	Epsilon  float32

	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func NewRegressor(inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts RegressorOpts) *Regressor {
	nn := godl.NewModel()

	embedder := godl.EmbeddingGenerator(nn, inputDim, catDims, catIdxs, catEmbDim, godl.EmbeddingOpts{
		WeightsInit: opts.WeightsInit,
	})

	embedDimSum := 0
	for _, v := range catEmbDim {
		embedDimSum += v
	}

	tabNetInputDim := inputDim + embedDimSum - len(catEmbDim)
	tn := TabNet(nn, TabNetOpts{
		OutputSize:         1,
		BatchSize:          opts.BatchSize,
		VirtualBatchSize:   opts.VirtualBatchSize,
		InputSize:          tabNetInputDim,
		MaskFunction:       gorgonia.Sigmoid,
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

	layer := godl.Sequential(nn, embedder, tn)

	return &Regressor{
		model: nn,
		layer: layer,
	}
}

func (r *Regressor) Model() *godl.Model {
	return r.model
}

func (r *Regressor) Train(trainX, trainY, validateX, validateY tensor.Tensor, opts godl.TrainOpts) error {
	if opts.CostFn == nil {
		lambdaSparse := gorgonia.NewConstant(float32(1e-3))
		opts.CostFn = func(output *gorgonia.Node, innerLoss *gorgonia.Node, y *gorgonia.Node) *gorgonia.Node {
			cost := godl.MSELoss(output, y, godl.MSELossOpts{})

			// r.model.Watch("output", gorgonia.Must(gorgonia.Sum(output)))
			// r.model.Watch("loss", cost)
			// r.model.Watch("innerLoss", innerLoss)

			cost = gorgonia.Must(gorgonia.Sub(cost, gorgonia.Must(gorgonia.Mul(lambdaSparse, innerLoss))))

			return cost
		}
	}

	if opts.Solver == nil {
		opts.Solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)), gorgonia.WithLearnRate(0.02), gorgonia.WithClip(1.0))
		// opts.Solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)), gorgonia.WithLearnRate(0.02))
	}

	return r.model.Train(r.layer, trainX, trainY, validateX, validateY, opts)
}
