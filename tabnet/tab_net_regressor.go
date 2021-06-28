package tabnet

import (
	"log"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Regressor struct {
	opts  RegressorOpts
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

func newModel(training bool, batchSize int, inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts RegressorOpts) (*godl.Model, godl.Layer) {
	nn := godl.NewModel()
	nn.Training = training

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
		BatchSize:          batchSize,
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

	return nn, layer
}

func NewRegressor(inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts RegressorOpts) *Regressor {
	train, trainLayer := newModel(true, opts.BatchSize, inputDim, catDims, catIdxs, catEmbDim, opts)

	return &Regressor{
		opts:  opts,
		model: train,
		layer: trainLayer,
	}
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
		// opts.Solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)), gorgonia.WithLearnRate(0.001), gorgonia.WithClip(1.0))
		opts.Solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)), gorgonia.WithLearnRate(1e-3))
	}

	return godl.Train(r.model, r.layer, trainX, trainY, validateX, validateY, opts)
}

// FIXME: this shouldn't receive Y
func (r *Regressor) Solve(x tensor.Tensor, y tensor.Tensor) (tensor.Tensor, error) {
	predictor, err := r.model.Predictor(r.layer, godl.PredictOpts{
		InputShape: tensor.Shape{r.opts.BatchSize, x.Shape()[1]},
	})
	if err != nil {
		return nil, err
	}

	yPos := 0
	correct := 0.0

	godl.InBatches(x, r.opts.BatchSize, func(v tensor.Tensor) {
		val, err := predictor(v)
		if err != nil {
			panic(err)
		}

		t := val.(tensor.Tensor)

		log.Printf("output: %v", t.Shape())

		for _, o := range t.Data().([]float32) {
			yVal, err := y.At(yPos, 0)
			if err != nil {
				panic(err)
			}

			// log.Printf("%v == %v", yVal, o)
			if yVal.(float32) == 1 {
				if o > 0.5 {
					correct++
				}
			} else {
				if o <= 0.5 {
					correct++
				}
			}

			yPos++
		}
	})

	log.Printf("r=%v", correct/float64(yPos))

	return nil, nil
}
