package godl

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/dcu/godl/storage"
	"gonum.org/v1/plot/vg"
	"gorgonia.org/gorgonia"
	"gorgonia.org/qol/plot"
	"gorgonia.org/tensor"
)

const (
	heatmapPath     = "heatmap"
	bufferSizeModel = 16
)

// Model implements the tab net model
type Model struct {
	g          *gorgonia.ExprGraph
	learnables gorgonia.Nodes
	watchables []watchable

	Training bool

	Storage *storage.Storage
}

// NewModel creates a new model for the neural network
func NewModel() *Model {
	return &Model{
		g:          gorgonia.NewGraph(),
		learnables: make([]*gorgonia.Node, 0, bufferSizeModel),
		watchables: make([]watchable, 0),
		Storage:    storage.NewStorage(),
	}
}

// WriteSVG creates a SVG representation of the node
func (m *Model) WriteSVG(path string) error {
	b := m.g.ToDot()

	fileName := "graph.dot"

	err := ioutil.WriteFile(fileName, []byte(b), 0644)
	if err != nil {
		return err
	}

	defer func() { _ = os.Remove(fileName) }()

	cmd := exec.Command("dot", "-T", "svg", fileName, "-o", path)

	return cmd.Run()
}

// ExprGraph returns the graph for the model
func (m *Model) ExprGraph() *gorgonia.ExprGraph {
	return m.g
}

// Learnables returns all learnables in the model
func (m *Model) Learnables() gorgonia.Nodes {
	return m.learnables
}

// Run runs the virtual machine in prediction mode
func (m *Model) Run(vmOpts ...gorgonia.VMOpt) error {
	vm := gorgonia.NewTapeMachine(m.g, vmOpts...)

	err := vm.RunAll()
	if err != nil {
		return err
	}

	return vm.Close()
}

type PredictOpts struct {
	InputShape tensor.Shape
	DevMode    bool
}

type Predictor func(x tensor.Tensor) (gorgonia.Value, error)

func (o *PredictOpts) setDefaults() {
	if o.InputShape == nil {
		panic("InputShape is required")
	}
}

func (m *Model) Predictor(layer Layer, opts PredictOpts) (Predictor, error) {
	opts.setDefaults()

	x := gorgonia.NewTensor(
		m.g,
		tensor.Float64,
		opts.InputShape.Dims(),
		gorgonia.WithName("input"),
		gorgonia.WithShape(opts.InputShape...),
	)

	result, err := layer(x)
	if err != nil {
		return nil, fmt.Errorf("error running layer: %w", err)
	}

	vmOpts := []gorgonia.VMOpt{
		gorgonia.EvalMode(),
	}

	if opts.DevMode {
		vmOpts = append(
			vmOpts,
			gorgonia.TraceExec(),
			gorgonia.WithInfWatch(),
			gorgonia.WithNaNWatch(),
		)
	}

	var predVal gorgonia.Value

	gorgonia.Read(result.Output, &predVal)

	return func(input tensor.Tensor) (gorgonia.Value, error) {
		gorgonia.Let(x, input)

		if err := m.Run(vmOpts...); err != nil {
			return nil, fmt.Errorf("failed to run prediction: %w", err)
		}

		return predVal, nil
	}, nil
}

func (m Model) saveHeatmaps(epoch, batch, batchSize, features int) {
	for _, v := range m.learnables {
		wt := v.Value().(tensor.Tensor)
		wtShape := wt.Shape().Clone()
		x, y := wtShape[0], tensor.Shape(wtShape[1:]).TotalSize()

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

// CheckArity checks if the arity is the correct one
func (m Model) CheckArity(lt LayerType, nodes []*gorgonia.Node, arity int) error {
	if len(nodes) != arity {
		return ErrorF(lt, "arity doesn't match, expected %d, got %d", arity, len(nodes))
	}

	return nil
}
