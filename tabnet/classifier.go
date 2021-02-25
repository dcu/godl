package tabnet

import (
	"github.com/dcu/deepzen"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Classifier struct {
	model *deepzen.Model
	layer deepzen.Layer
}

type ClassifierOpts struct {
	BatchSize        int
	VirtualBatchSize int
	MaskFunction     deepzen.ActivationFn
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

func NewClassifier(inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts ClassifierOpts) *Classifier {
	nn := deepzen.NewModel()

	embedder := deepzen.EmbeddingGenerator(nn, inputDim, catDims, catIdxs, catEmbDim, deepzen.EmbeddingOpts{
		WeightsInit: opts.WeightsInit,
	})

	embedDimSum := 0
	for _, v := range catEmbDim {
		embedDimSum += v
	}

	tabNetInputDim := inputDim + embedDimSum - len(catEmbDim)
	tn := TabNet(nn, TabNetOpts{
		OutputDimension:    1,
		BatchSize:          opts.BatchSize,
		VirtualBatchSize:   opts.VirtualBatchSize,
		InputDimension:     tabNetInputDim,
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

	layer := deepzen.Sequential(nn, embedder, tn)

	return &Classifier{
		model: nn,
		layer: layer,
	}
}

func (r *Classifier) Model() *deepzen.Model {
	return r.model
}

func (r *Classifier) Train(trainX, trainY, validateX, validateY tensor.Tensor, opts deepzen.TrainOpts) error {
	if opts.CostFn == nil {
		lambdaSparse := gorgonia.NewConstant(float32(1e-3))
		opts.CostFn = func(output *gorgonia.Node, innerLoss *gorgonia.Node, y *gorgonia.Node) *gorgonia.Node {
			cost := deepzen.CrossEntropyLoss(output, y, deepzen.CrossEntropyLossOpt{})
			cost = gorgonia.Must(gorgonia.Sub(cost, gorgonia.Must(gorgonia.Mul(lambdaSparse, innerLoss))))

			return cost
		}
	}

	if opts.Solver == nil {
		opts.Solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)), gorgonia.WithLearnRate(0.02), gorgonia.WithClip(1.0))
	}

	return r.model.Train(r.layer, trainX, trainY, validateX, validateY, opts)
}
