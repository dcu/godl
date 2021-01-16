package tabnet

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const bufferSizeModel = 16

// TrainOpts are the options to train the model
type TrainOpts struct {
	Epochs    int
	BatchSize int

	CostFn func(output *gorgonia.Node, loss *gorgonia.Node, y *gorgonia.Node) *gorgonia.Node
}

func (o *TrainOpts) setDefaults() {
	if o.Epochs == 0 {
		o.Epochs = 10
	}

	if o.BatchSize == 0 {
		o.BatchSize = 1024
	}
}

// Model implements the tab net model
type Model struct {
	g          *gorgonia.ExprGraph
	learnables gorgonia.Nodes

	model map[string]gorgonia.Value
}

// NewModel creates a new model for the neural network
func NewModel() *Model {
	return &Model{
		g:          gorgonia.NewGraph(),
		learnables: make([]*gorgonia.Node, 0, bufferSizeModel),
		model:      make(map[string]gorgonia.Value, bufferSizeModel),
	}
}

func (m *Model) ExprGraph() *gorgonia.ExprGraph {
	return m.g
}

func (m *Model) Train(layer Layer, trainX tensor.Tensor, trainY tensor.Tensor, opts TrainOpts) error {
	opts.setDefaults()

	numExamples, features := trainX.Shape()[0], trainX.Shape()[1]
	batches := numExamples / opts.BatchSize

	x := gorgonia.NewTensor(m.g, tensor.Float64, trainX.Shape().Dims(), gorgonia.WithShape(opts.BatchSize, features), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(m.g, tensor.Float64, gorgonia.WithShape(opts.BatchSize, trainY.Shape()[1]), gorgonia.WithName("y"))

	output, loss, err := layer(x)
	if err != nil {
		return fmt.Errorf("error running layer: %w", err)
	}

	if loss == nil {
		return fmt.Errorf("loss must be returned in training mode")
	}

	var (
		costVal gorgonia.Value
		predVal gorgonia.Value
	)

	{
		cost := opts.CostFn(output, loss, y)

		gorgonia.Read(cost, &costVal)
		gorgonia.Read(output, &predVal)

		if _, err := gorgonia.Grad(cost, m.learnables...); err != nil {
			return fmt.Errorf("error calculating gradient: %w", err)
		}
	}

	// logFile, _ := os.Create("log")
	// defer logFile.Close()

	vm := gorgonia.NewTapeMachine(m.g,
		gorgonia.BindDualValues(m.learnables...),
		gorgonia.TraceExec(),
		// gorgonia.WithLogger(log.New(logFile, "[g]", log.LstdFlags)),
		// gorgonia.WithWatchlist(),
		gorgonia.WithNaNWatch(),
		gorgonia.WithInfWatch(),
	)
	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)))

	defer vm.Close()

	// bar := pb.New(batches)
	// bar.SetRefreshRate(time.Second)
	// bar.SetMaxWidth(80)

	for i := 0; i < opts.Epochs; i++ {
		// bar.Prefix(fmt.Sprintf("Epoch %d", i))
		// bar.Set(0)
		// bar.Start()

		for b := 0; b < batches; b++ {
			start := b * opts.BatchSize
			end := start + opts.BatchSize

			if start >= numExamples {
				break
			}

			if end > numExamples {
				end = numExamples
			}

			xVal, err := trainX.Slice(gorgonia.S(start, end))
			if err != nil {
				return err
			}

			yVal, err := trainY.Slice(gorgonia.S(start, end))
			if err != nil {
				return err
			}

			err = xVal.(*tensor.Dense).Reshape(opts.BatchSize, features)
			if err != nil {
				return err
			}

			err = gorgonia.Let(x, xVal)
			if err != nil {
				log.Fatalf("error assigning x: %v", err)
			}

			err = gorgonia.Let(y, yVal)
			if err != nil {
				log.Fatalf("error assigning y: %v", err)
			}

			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
			}

			if err = solver.Step(gorgonia.NodesToValueGrads(m.learnables)); err != nil {
				log.Fatalf("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, b, err)
			}

			fmt.Printf(" Epoch %d %d | cost %v\n", i, b, costVal)

			vm.Reset()
			// bar.Increment()
		}

		// fmt.Printf(" Epoch %d | cost %v", i, costVal)
	}

	fmt.Println("")

	return nil
}

func (m Model) checkArity(contextName string, nodes []*gorgonia.Node, arity int) error {
	if len(nodes) != arity {
		return fmt.Errorf("arity doesn't match on %s, expected %d, got %d", contextName, arity, len(nodes))
	}

	return nil
}
