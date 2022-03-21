package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/dcu/godl"
	"github.com/dcu/godl/table"
	"github.com/dcu/godl/tabnet"
	"gorgonia.org/gorgonia"
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

func main() {
	p, err := table.ReadCSV("adult.data")
	handleErr(err)

	fmt.Printf(">> Uniq values per column\n")
	for col, classes := range p.ClassesByColumn {
		if len(classes) > 0 {
			fmt.Printf("%s: %d\n", p.Header[col], len(classes))
		}
	}

	p.AddTag(table.RandValueIn(map[string]float64{
		"train":    0.8,
		"validate": 0.1,
		"test":     0.1,
	}))

	trainX, trainY := p.ToTensors(table.ToTensorOpts{TargetColumns: []int{14}, SelectTags: []string{"train"}})
	validateX, validateY := p.ToTensors(table.ToTensorOpts{TargetColumns: []int{14}, SelectTags: []string{"validate"}})
	testX, testY := p.ToTensors(table.ToTensorOpts{TargetColumns: []int{14}, SelectTags: []string{"test"}})

	log.Printf("train x: %v train y: %v", trainX.Shape(), trainY.Shape())
	log.Printf("validateX: %v validateY: %v", validateX.Shape(), validateY.Shape())

	catIdxs, catDims := p.CategoricalColumns(14)

	batchSize := 1024
	virtualBatchSize := 128
	catEmbDim := []int{5, 4, 3, 6, 2, 2, 1, 10}

	log.Printf("cat dims: %v", catDims)
	log.Printf("cat emb dims: %v", catEmbDim)
	log.Printf("cat idxs: %v", catIdxs)

	regressor := tabnet.NewRegressor(
		trainX.Shape()[1], catDims, catIdxs, catEmbDim, tabnet.RegressorOpts{
			BatchSize:          batchSize,
			VirtualBatchSize:   virtualBatchSize,
			MaskFunction:       godl.Sparsemax,
			PredictionLayerDim: 8,
			AttentionLayerDim:  8,
			Gamma:              1.3,
			DecisionSteps:      3,
			IndependentBlocks:  2,
			SharedBlocks:       2,
			Momentum:           0.02,
			WithBias:           false,
			Epsilon:            1e-15,
		},
	)

	err = regressor.Train(trainX, trainY, validateX, validateY, godl.TrainOpts{
		BatchSize: batchSize,
		Epochs:    1,
		DevMode:   true,
		Solver:    gorgonia.NewAdamSolver(gorgonia.WithLearnRate(2e-2)),
		MatchTypeFor: func(predVal, targetVal []float32) godl.MatchType {
			log.Printf("%v vs %v", predVal, targetVal)

			if targetVal[0] == 1 {
				if predVal[0] >= 0.5 {
					return godl.MatchTypeTruePositive
				} else {
					return godl.MatchTypeFalsePositive
				}
			} else { // == 0
				if predVal[0] < 0.5 {
					return godl.MatchTypeTrueNegative
				} else {
					return godl.MatchTypeFalseNegative
				}
			}
		},
		ValidationObserver: func(confMat godl.ConfusionMatrix, cost float32) {
			fmt.Printf("%v\nCost: %0.4f", confMat, cost)
		},
		// WithLearnablesHeatmap: true,
	})
	handleErr(err)

	out, err := regressor.Solve(testX, testY)
	handleErr(err)

	log.Printf("out: %v", out)
}
