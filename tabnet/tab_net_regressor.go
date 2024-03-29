package tabnet

import (
	"log"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Regressor struct {
	opts   RegressorOpts
	model  *godl.Model
	tabnet *TabNetModule
}

type RegressorOpts struct {
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

func newModel(training bool, batchSize int, inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts RegressorOpts) (*godl.Model, *TabNetModule) {
	nn := godl.NewModel()

	layer := TabNet(nn, TabNetOpts{
		OutputSize:         1,
		BatchSize:          batchSize,
		VirtualBatchSize:   opts.VirtualBatchSize,
		InputSize:          inputDim,
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
		CatDims:            catDims,
		CatIdxs:            catIdxs,
		CatEmbDim:          catEmbDim,
	})

	return nn, layer
}

func NewRegressor(inputDim int, catDims []int, catIdxs []int, catEmbDim []int, opts RegressorOpts) *Regressor {
	model, layer := newModel(true, opts.BatchSize, inputDim, catDims, catIdxs, catEmbDim, opts)

	return &Regressor{
		opts:   opts,
		model:  model,
		tabnet: layer,
	}
}

func (r *Regressor) Train(trainX, trainY, validateX, validateY tensor.Tensor, opts godl.TrainOpts) error {
	log.Printf("input: %v", trainX)

	if opts.CostFn == nil {
		lambdaSparse := gorgonia.NewConstant(float32(1e-3), gorgonia.WithName("LambdaSparse"))
		mseLoss := godl.MSELoss(godl.MSELossOpts{})

		opts.CostFn = func(output godl.Nodes, target *godl.Node) *gorgonia.Node {
			cost := mseLoss(output, target)
			tmpLoss := gorgonia.Must(gorgonia.Mul(output[1], lambdaSparse))
			cost = gorgonia.Must(gorgonia.Sub(cost, tmpLoss))

			return cost
		}
	}

	if opts.Solver == nil {
		opts.Solver = gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)), gorgonia.WithLearnRate(0.02))
	}

	return godl.Train(r.model, r.tabnet, trainX, trainY, validateX, validateY, opts)
}

// FIXME: this shouldn't receive Y
func (r *Regressor) Solve(x tensor.Tensor, y tensor.Tensor) (tensor.Tensor, error) {
	predictor, err := r.model.Predictor(r.tabnet, godl.PredictOpts{
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
