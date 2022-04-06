package godl

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"time"

	"github.com/fatih/color"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TrainOpts are the options to train the model
type TrainOpts struct {
	Epochs    int
	BatchSize int

	// DevMode detects common issues like exploding and vanishing gradients at the cost of performance
	DevMode bool

	WriteGraphFileTo string

	// WithLearnablesHeatmap writes images representing heatmaps for the weights. Use it to debug.
	WithLearnablesHeatmap bool

	// Solver defines the solver to use. It uses gorgonia.AdamSolver by default if none is passed
	Solver gorgonia.Solver

	// ValidateEvery indicates the number of epochs to run before running a validation. Defaults 1 (every epoch)
	ValidateEvery int

	CostObserver       func(epoch int, totalEpoch, batch int, totalBatch int, cost float32)
	ValidationObserver func(confMat ConfusionMatrix, cost float32)
	MatchTypeFor       func(predVal, targetVal []float32) MatchType
	CostFn             CostFn
}

func (o *TrainOpts) setDefaults() {
	if o.Epochs == 0 {
		o.Epochs = 10
	}

	if o.BatchSize == 0 {
		o.BatchSize = 1024
	}

	if o.ValidateEvery == 0 {
		o.ValidateEvery = 1
	}

	if o.CostFn == nil {
		panic("CostFN must be set")
	}
}

// Train trains the model with the given data
func Train(m *Model, module Module, trainX, trainY, validateX, validateY tensor.Tensor, opts TrainOpts) error {
	opts.setDefaults()

	if opts.DevMode {
		warn("Start training in dev mode")

		defer func() {
			if err := recover(); err != nil {
				graphFileName := "graph.dot"

				log.Printf("panic triggered, dumping the model graph to: %v", graphFileName)
				_ = ioutil.WriteFile(graphFileName, []byte(m.trainGraph.ToDot()), 0644)
				panic(err)
			}
		}()
	}

	if opts.WithLearnablesHeatmap {
		warn("Heatmaps will be stored in: %s", heatmapPath)
		_ = os.RemoveAll(heatmapPath)
	}

	dl := NewDataLoader(trainX, trainY, DataLoaderOpts{
		BatchSize: opts.BatchSize,
		Shuffle:   false,
	})

	xShape := append(tensor.Shape{opts.BatchSize}, trainX.Shape()[1:]...)

	x := gorgonia.NewTensor(m.trainGraph, tensor.Float32, trainX.Shape().Dims(), gorgonia.WithShape(xShape...), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(m.trainGraph, tensor.Float32, gorgonia.WithShape(opts.BatchSize, trainY.Shape()[1]), gorgonia.WithName("y"))

	result := module.Forward(x)

	if opts.WriteGraphFileTo != "" {
		m.WriteSVG(opts.WriteGraphFileTo)
	}

	var (
		costVal gorgonia.Value
		predVal gorgonia.Value
	)

	{
		cost := opts.CostFn(result, y)

		gorgonia.Read(cost, &costVal)
		gorgonia.Read(result[0], &predVal)

		if _, err := gorgonia.Grad(cost, m.Learnables()...); err != nil {
			return fmt.Errorf("error calculating gradient: %w", err)
		}
	}

	validationGraph := m.trainGraph.SubgraphRoots(result[0])
	validationGraph.RemoveNode(y)

	m.evalGraph = validationGraph

	vmOpts := []gorgonia.VMOpt{
		gorgonia.BindDualValues(m.learnables...),
	}

	if opts.DevMode {
		vmOpts = append(
			vmOpts,
			gorgonia.TraceExec(),
			gorgonia.WithNaNWatch(),
			gorgonia.WithInfWatch(),
		)
	}

	vm := gorgonia.NewTapeMachine(m.trainGraph, vmOpts...)

	if opts.Solver == nil {
		info("defaulting to RMS solver")

		opts.Solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)))
	}

	defer vm.Close()

	startTime := time.Now()

	for i := 0; i < opts.Epochs; i++ {
		for dl.HasNext() {
			xVal, yVal := dl.Next()

			err := gorgonia.Let(x, xVal)
			if err != nil {
				fatal("error assigning x: %v", err)
			}

			err = gorgonia.Let(y, yVal)
			if err != nil {
				fatal("error assigning y: %v", err)
			}

			if err = vm.RunAll(); err != nil {
				fatal("Failed at epoch  %d, batch %d. Error: %v", i, dl.CurrentBatch, err)
			}

			if opts.WithLearnablesHeatmap {
				m.saveHeatmaps(i, dl.CurrentBatch, dl.opts.BatchSize, dl.FeaturesShape[0])
			}

			if err = opts.Solver.Step(gorgonia.NodesToValueGrads(m.learnables)); err != nil {
				fatal("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, dl.CurrentBatch, err)
			}

			if opts.CostObserver != nil {
				opts.CostObserver(i, opts.Epochs, dl.CurrentBatch, dl.Batches, costVal.Data().(float32))
			} else {
				// color.Yellow(" Epoch %d %d | cost %v (%v)\n", i, dl.CurrentBatch, costVal, time.Since(startTime))
			}

			m.PrintWatchables()

			vm.Reset()
		}

		dl.Reset()

		if i%opts.ValidateEvery == 0 {
			err := Validate(m, x, y, costVal, predVal, validateX, validateY, opts)
			if err != nil {
				color.Red("Failed to run validation on epoch %v: %v", i, err)
			}

			color.Yellow(" Epoch %d | cost %v (%v)\n", i, costVal, time.Since(startTime))
		}
	}

	return nil
}
