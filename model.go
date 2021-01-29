package tabnet

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/fatih/color"
	"gonum.org/v1/plot/vg"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/encoding/dot"
	"gorgonia.org/qol/plot"
	"gorgonia.org/tensor"
)

const (
	heatmapPath     = "heatmap"
	bufferSizeModel = 16
)

// TrainOpts are the options to train the model
type TrainOpts struct {
	Epochs    int
	BatchSize int

	DevMode               bool
	WithLearnablesHeatmap bool

	// Solver defines the solver to use. It uses gorgonia.AdamSolver by default if none is passed
	Solver gorgonia.Solver

	CostObserver func(epoch int, totalEpoch, batch int, totalBatch int, cost float32)
	CostFn       func(output *gorgonia.Node, loss *gorgonia.Node, y *gorgonia.Node) *gorgonia.Node
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
	watchables []watchable

	model map[string]gorgonia.Value
}

// NewModel creates a new model for the neural network
func NewModel() *Model {
	return &Model{
		g:          gorgonia.NewGraph(),
		learnables: make([]*gorgonia.Node, 0, bufferSizeModel),
		watchables: make([]watchable, 0),
		model:      make(map[string]gorgonia.Value, bufferSizeModel),
	}
}

// ToSVG creates a SVG representation of the node
func (m *Model) ToSVG(path string) error {
	b, err := dot.Marshal(m.g)
	if err != nil {
		return err
	}

	fileName := "graph.dot"

	err = ioutil.WriteFile(fileName, b, 0644)
	if err != nil {
		return err
	}

	defer func() { _ = os.Remove(fileName) }()

	cmd := exec.Command("dot", "-T", "svg", fileName, "-o", path)

	return cmd.Run()
}

func (m *Model) ExprGraph() *gorgonia.ExprGraph {
	return m.g
}

func (m *Model) Train(layer Layer, trainX, trainY, validateX, validateY tensor.Tensor, opts TrainOpts) error {
	opts.setDefaults()

	if opts.DevMode {
		warn("Start training in dev mode")

		defer func() {
			if err := recover(); err != nil {
				graphFileName := "graph.dot"

				log.Printf("panic triggered, dumping the model graph to: %v", graphFileName)
				_ = ioutil.WriteFile(graphFileName, []byte(m.g.ToDot()), 0644)
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
	})

	x := gorgonia.NewTensor(m.g, tensor.Float32, trainX.Shape().Dims(), gorgonia.WithShape(opts.BatchSize, trainX.Shape()[1]), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(m.g, tensor.Float32, gorgonia.WithShape(opts.BatchSize, trainY.Shape()[1]), gorgonia.WithName("y"))

	result, err := layer(x)
	if err != nil {
		return fmt.Errorf("error running layer: %w", err)
	}

	var (
		costVal gorgonia.Value
		predVal gorgonia.Value
	)

	{
		cost := opts.CostFn(result.Output, result.Loss, y)

		gorgonia.Read(cost, &costVal)
		gorgonia.Read(result.Output, &predVal)

		if _, err := gorgonia.Grad(cost, m.learnables...); err != nil {
			return fmt.Errorf("error calculating gradient: %w", err)
		}
	}

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

	vm := gorgonia.NewTapeMachine(m.g, vmOpts...)

	if opts.Solver == nil {
		info("defaulting to RMS solver")

		opts.Solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)))
	}

	defer vm.Close()

	startTime := time.Now()

	for i := 0; i < opts.Epochs; i++ {
		for dl.HasNext() {
			xVal, yVal := dl.Next()

			err = gorgonia.Let(x, xVal)
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
				color.Yellow(" Epoch %d %d | cost %v\n", i, dl.CurrentBatch, costVal)
			}

			m.PrintWatchables()

			vm.Reset()
		}

		dl.Reset()

		_ = startTime
		if opts.CostObserver != nil {
			opts.CostObserver(i+1, opts.Epochs, dl.Batches, dl.Batches, costVal.Data().(float32))
		} else {
			fmt.Printf(" Epoch %d | cost %v (%v)\n", i, costVal, time.Since(startTime))
		}
	}

	return m.validate(x, y, costVal, predVal, validateX, validateY, opts)
}

func (m *Model) validate(x, y *gorgonia.Node, costVal, predVal gorgonia.Value, validateX, validateY tensor.Tensor, opts TrainOpts) error {
	opts.setDefaults()

	numExamples, features := validateX.Shape()[0], validateX.Shape()[1]
	batches := numExamples / opts.BatchSize

	vm := gorgonia.NewTapeMachine(m.g)

	if opts.Solver == nil {
		opts.Solver = gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(opts.BatchSize)))
	}

	defer vm.Close()

	correct := 0

	for b := 0; b < batches; b++ {
		start := b * opts.BatchSize
		end := start + opts.BatchSize

		if start >= numExamples {
			break
		}

		if end > numExamples {
			end = numExamples
		}

		xVal, err := validateX.Slice(gorgonia.S(start, end))
		if err != nil {
			return err
		}

		yVal, err := validateY.Slice(gorgonia.S(start, end))
		if err != nil {
			return err
		}

		err = xVal.(*tensor.Dense).Reshape(opts.BatchSize, features)
		if err != nil {
			return err
		}

		err = gorgonia.Let(x, xVal)
		if err != nil {
			fatal("error assigning x: %v", err)
		}

		err = gorgonia.Let(y, yVal)
		if err != nil {
			fatal("error assigning y: %v", err)
		}

		if err = vm.RunAll(); err != nil {
			fatal("Failed batch %d. Error: %v", b, err)
		}

		if opts.CostObserver != nil {
			opts.CostObserver(1, 1, b+1, batches, costVal.Data().(float32))
		} else {
			color.Yellow(" Validation cost %v\n", costVal)
		}

		pred := predVal.Data().([]float32)

		for j := 0; j < opts.BatchSize; j++ {
			targetVal, err := yVal.Slice(gorgonia.S(j))
			if err != nil {
				panic(err)
			}

			target := targetVal.Data().(float32)

			if target == 1 {
				if pred[j] >= 0.5 {
					correct++
				}
			} else {
				if pred[j] < 0.5 {
					correct++
				}
			}
		}

		vm.Reset()
	}

	accuracy := float64(correct) / float64(numExamples)

	log.Printf("accuracy: %0.3f%%", accuracy*100)

	return nil
}

func (m Model) saveHeatmaps(epoch, batch, batchSize, features int) {
	for _, v := range m.learnables {
		wt := v.Value().(tensor.Tensor)
		wtShape := wt.Shape().Clone()
		x, y := wtShape[0], tensor.Shape(wtShape[1:]).TotalSize()

		if x == 1 {
			x, y = PrimeFactors(y)
		}

		newShape := tensor.Shape{x, y}

		grad, err := v.Grad()
		if err != nil {
			panic(err)
		}

		gradT := grad.(tensor.Tensor)

		pathName := filepath.Join(heatmapPath, v.Name())
		fileName := fmt.Sprintf("%s/%d_%d_%v.png", pathName, epoch, batch, wtShape)
		gradFileName := fmt.Sprintf("%s/grad_%d_%d_%v.png", pathName, epoch, batch, wtShape)

		err = wt.Reshape(newShape...)
		if err != nil {
			panic(err)
		}

		p, err := plot.Heatmap(wt, nil)
		if err != nil {
			panic(fmt.Errorf("failed to process %s: %w", fileName, err))
		}

		err = gradT.Reshape(newShape...)
		if err != nil {
			panic(err)
		}

		pGrad, err := plot.Heatmap(gradT, nil)
		if err != nil {
			panic(fmt.Errorf("failed to process %s: %w", fileName, err))
		}

		err = wt.Reshape(wtShape...)
		if err != nil {
			panic(err)
		}

		err = gradT.Reshape(wtShape...)
		if err != nil {
			panic(err)
		}

		width := vg.Length(newShape[0]) * vg.Centimeter
		height := vg.Length(newShape[1]) * vg.Centimeter

		_ = os.MkdirAll(pathName, 0755)
		_ = p.Save(width, height, fileName)

		_ = pGrad.Save(width, height, gradFileName)
	}
}

func (m Model) CheckArity(lt LayerType, nodes []*gorgonia.Node, arity int) error {
	if len(nodes) != arity {
		return errorF(lt, "arity doesn't match, expected %d, got %d", arity, len(nodes))
	}

	return nil
}

// PrimeFactors Get all prime factors of a given number n
func PrimeFactors(n int) (int, int) {
	pfs := make([]int, 0)

	// Get the number of 2s that divide n
	for n%2 == 0 {
		pfs = append(pfs, 2)
		n = n / 2
	}

	// n must be odd at this point. so we can skip one element
	// (note i = i + 2)
	for i := 3; i*i <= n; i = i + 2 {
		// while i divides n, append i and divide n
		for n%i == 0 {
			pfs = append(pfs, i)
			n = n / i
		}
	}

	// This condition is to handle the case when n is a prime number
	// greater than 2
	if n > 2 {
		pfs = append(pfs, n)
	}

	mul := func(arr []int) int {
		r := 1
		for _, v := range arr {
			r *= v
		}

		return r
	}

	first := mul(pfs[:len(pfs)/2])
	second := mul(pfs[len(pfs)/2:])

	return first, second
}
