package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/dcu/tabnet"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func handleErr(err error) {
	if err == nil {
		return
	}

	panic(err)
}

type Processor struct {
	columns                int
	categoricalColumnsMap  []map[string]float64
	columnNames            []string
	categoricalColumnsUniq []map[string]int

	trainingRows int
	validateRows int

	trainX []float64
	trainY []float64

	validateX []float64
	validateY []float64
}

func newProcessor(classes int) *Processor {
	return &Processor{
		columns:               classes,
		categoricalColumnsMap: make([]map[string]float64, classes),
	}
}

func (p Processor) catDims() []int {
	dims := make([]int, 0, len(p.categoricalColumnsUniq))

	for _, v := range p.categoricalColumnsUniq {
		if len(v) != 0 {
			dims = append(dims, len(v))
		}
	}

	return dims
}

func (p Processor) catIdxs() []int {
	indexes := make([]int, 0, len(p.categoricalColumnsUniq))

	for i, v := range p.categoricalColumnsUniq {
		if len(v) != 0 {
			indexes = append(indexes, i)
		}
	}

	return indexes
}

func (p *Processor) assignID(categoricalColumnPos int, categoricalColumnValue string) float64 {
	// fmt.Printf("assign id for %d %v\n", categoricalColumnPos, categoricalColumnValue)

	m := p.categoricalColumnsMap[categoricalColumnPos]
	if m == nil {
		m = make(map[string]float64, 64)
		p.categoricalColumnsMap[categoricalColumnPos] = m
	}

	id, ok := m[categoricalColumnValue]
	if ok {
		return id
	}

	id = float64(len(p.categoricalColumnsMap[categoricalColumnPos]))
	m[categoricalColumnValue] = id

	return id
}

func (p *Processor) processRow(record []string, targetCol int, targetVal string) {
	x := make([]float64, p.columns)
	y := 0.0

	if p.columnNames == nil {
		p.columnNames = record
	}

	if p.categoricalColumnsUniq == nil {
		p.categoricalColumnsUniq = make([]map[string]int, p.columns)
		for i := 0; i < p.columns; i++ {
			p.categoricalColumnsUniq[i] = make(map[string]int, 4096)
		}
	}

	colPos := 0

	for i, r := range record {
		if i == targetCol {
			if r == targetVal {
				y = 1.0
			}

			continue
		}

		r = strings.TrimSpace(r)

		if r == "" {
			panic(fmt.Errorf("empty column in record: %v", record))
		}

		value, ok := parseNumber(r)
		if ok {
			x[i] = value
		} else if i != targetCol {
			// categorical column
			x[i] = p.assignID(colPos, r)

			p.categoricalColumnsUniq[colPos][r]++
		}

		colPos++
	}

	if rand.Float64() <= 0.8 {
		p.trainX = append(p.trainX, x...)
		p.trainY = append(p.trainY, y)

		p.trainingRows++
	} else {
		p.validateX = append(p.validateX, x...)
		p.validateY = append(p.validateY, y)

		p.validateRows++
	}
}

func (p *Processor) TrainingTensors() (x tensor.Tensor, y tensor.Tensor) {
	return tensor.New(
			tensor.WithShape(p.trainingRows, p.columns),
			tensor.WithBacking(p.trainX),
		),
		tensor.New(
			tensor.WithShape(p.trainingRows, 1),
			tensor.WithBacking(p.trainY),
		)
}

func parseNumber(v string) (float64, bool) {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		i, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return 0.0, false
		}

		return float64(i), true
	}

	return f, true
}

func process(filePath string) *Processor {
	f, err := os.Open(filePath)
	handleErr(err)

	defer func() { _ = f.Close() }()

	csvReader := csv.NewReader(f)
	processor := newProcessor(14)

	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}

		handleErr(err)

		processor.processRow(record, 14, ">50K")
	}

	return processor
}

func main() {
	p := process("adult.data")

	fmt.Printf(">> Uniq values per column\n")
	for col, uniqVals := range p.categoricalColumnsUniq {
		if len(uniqVals) > 0 {
			fmt.Printf("%s: %d\n", p.columnNames[col], len(uniqVals))
		}
	}

	trainX, trainY := p.TrainingTensors()

	log.Printf("train x: %v train y: %v", trainX.Shape(), trainY.Shape())

	batchSize := 1024

	model := tabnet.NewModel()

	catDims := p.catDims()
	catEmbDim := []int{5, 4, 3, 6, 2, 2, 1, 10}
	catIdxs := p.catIdxs()

	log.Printf("cat dims: %v", catDims)
	log.Printf("cat emb dims: %v", catEmbDim)
	log.Printf("cat idxs: %v", catIdxs)

	embedder := model.EmbeddingGenerator(14, catDims, catIdxs, catEmbDim, tabnet.EmbeddingOpts{})
	tn := model.TabNet(tabnet.TabNetOpts{
		OutputDimension:    1,
		PredictionLayerDim: 64,
		AttentionLayerDim:  64,
		BatchSize:          batchSize,
		InputDim:           39,
		MaskFunction:       gorgonia.Sigmoid,
	})

	layer := model.Sequential(embedder, tn)

	err := model.Train(layer, trainX, trainY, tabnet.TrainOpts{
		BatchSize: batchSize,
		Epochs:    3,
	})
	handleErr(err)
}
