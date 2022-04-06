package tabnet

import (
	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Classifier struct {
	model  *godl.Model
	tabnet *TabNetModule
}

type ClassifierOpts struct {
	BatchSize        int
	VirtualBatchSize int
	MaskFunction     activation.Function
	WithBias         bool

	SharedBlocks       int
	IndependentBlocks  int
	DecisionSteps      int
	PredictionLayerDim int
	AttentionLayerDim  int

	Gamma    float64
	Momentum float64
	Epsilon  float64

	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func NewClassifier(inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts ClassifierOpts) *Classifier {
	nn := godl.NewModel()

	tn := TabNet(nn, TabNetOpts{
		OutputSize:         1,
		BatchSize:          opts.BatchSize,
		VirtualBatchSize:   opts.VirtualBatchSize,
		InputSize:          inputDim,
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
		CatDims:            catDims,
		CatIdxs:            catIdxs,
		CatEmbDim:          catEmbDim,
	})

	return &Classifier{
		model:  nn,
		tabnet: tn,
	}
}

func (r *Classifier) Model() *godl.Model {
	return r.model
}

func (r *Classifier) Train(trainX, trainY, validateX, validateY tensor.Tensor, opts godl.TrainOpts) error {
	if opts.CostFn == nil {
		lambdaSparse := gorgonia.NewConstant(float32(1e-3))
		crossEntropy := godl.CategoricalCrossEntropyLoss(godl.CrossEntropyLossOpt{})

		opts.CostFn = func(output godl.Nodes, target *godl.Node) *gorgonia.Node {
			cost := crossEntropy(output, target)
			cost = gorgonia.Must(gorgonia.Sub(cost, gorgonia.Must(gorgonia.Mul(lambdaSparse, output[1]))))

			return cost
		}
	}

	if opts.Solver == nil {
		opts.Solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)), gorgonia.WithLearnRate(0.02))
	}

	return godl.Train(r.model, r.tabnet, trainX, trainY, validateX, validateY, opts)
}
