package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
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
	p, err := table.ReadCSV("dataset.csv")
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

	trainX, trainY := p.ToTensors(table.ToTensorOpts{TargetColumns: []int{10}, SelectTags: []string{"train"}})
	validateX, validateY := p.ToTensors(table.ToTensorOpts{TargetColumns: []int{10}, SelectTags: []string{"validate"}})
	testX, testY := p.ToTensors(table.ToTensorOpts{TargetColumns: []int{10}, SelectTags: []string{"test"}})

	log.Printf("rows: %v", len(p.Rows))

	log.Printf("train x: %v train y: %v", trainX, trainY)
	log.Printf("validateX: %v validateY: %v", validateX, validateY)
	log.Printf("testX: %v testY: %v", testX, testY)

	batchSize := 128
	if trainX.Shape()[0] < batchSize {
		batchSize = trainX.Shape()[0]
	}

	virtualBatchSize := 8

	regressor := tabnet.NewRegressor(
		trainX.Shape()[1], []int{}, []int{}, []int{}, tabnet.RegressorOpts{
			BatchSize:          batchSize,
			VirtualBatchSize:   virtualBatchSize,
			MaskFunction:       activation.Sigmoid,
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
		Epochs:    3,
		DevMode:   true,
		Solver:    gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.02), gorgonia.WithBatchSize(float64(batchSize))),
		MatchTypeFor: func(predVal, targetVal []float32) godl.MatchType {
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
		WithLearnablesHeatmap: false,
	})
	handleErr(err)

	out, err := regressor.Solve(testX, testY)
	handleErr(err)

	log.Printf("out: %v", out)
}
