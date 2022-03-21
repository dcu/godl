package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/dcu/godl"
	"github.com/dcu/godl/examples/mnist"
	"gorgonia.org/gorgonia"
)

var (
	datasetDir string
)

func init() {
	flag.StringVar(&datasetDir, "dataset-dir", "..", "The dir where the dataset is located")
}

func handleErr(what string, err error) {
	if err != nil {
		log.Panicf("%s: %v", what, err)
	}
}

func main() {
	trainX, trainY, err := mnist.Load(mnist.ModeTrain, datasetDir)
	handleErr("loading trainig mnist data", err)

	validateX, validateY, err := mnist.Load(mnist.ModeTrain, datasetDir)
	handleErr("loading validation mnist data", err)

	model := godl.NewModel()
	layer := godl.Sequential(
		model,
		godl.FC(model, godl.FCOpts{
			InputDimension:  784,
			OutputDimension: 300,
			WithBias:        true,
			Activation:      godl.Rectify,
		}),
		godl.BatchNorm1d(model, godl.BatchNormOpts{
			InputSize: 300,
		}),
		godl.FC(model, godl.FCOpts{
			InputDimension:  300,
			OutputDimension: 100,
			WithBias:        true,
			Activation:      godl.Rectify,
		}),
		godl.BatchNorm1d(model, godl.BatchNormOpts{
			InputSize: 100,
		}),
		godl.FC(model, godl.FCOpts{
			InputDimension:  100,
			OutputDimension: 10,
			WithBias:        true,
		}),
	)

	err = godl.Train(model, layer, trainX, trainY, validateX, validateY, godl.TrainOpts{
		Epochs:        3,
		ValidateEvery: 0,
		BatchSize:     64,
		// WriteGraphFileTo: "graph.svg",
		Solver: gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.001)),
		CostObserver: func(epoch, totalEpoch, batch, totalBatch int, cost float32) {
			// log.Printf("batch=%d/%d epoch=%d/%d cost=%0.3f", batch, totalBatch, epoch, totalEpoch, cost)
		},
		MatchTypeFor: func(predVal, targetVal []float32) godl.MatchType {
			var (
				rowLabel int
				yRowHigh float32
			)

			for k := 0; k < 10; k++ {
				if k == 0 {
					rowLabel = 0
					yRowHigh = targetVal[k]
				} else if targetVal[k] > yRowHigh {
					rowLabel = k
					yRowHigh = targetVal[k]
				}
			}

			var (
				rowGuess    int
				predRowHigh float32
			)

			for k := 0; k < 10; k++ {
				if k == 0 {
					rowGuess = 0
					predRowHigh = predVal[k]
				} else if predVal[k] > predRowHigh {
					rowGuess = k
					predRowHigh = predVal[k]
				}
			}

			if rowLabel == rowGuess {
				return godl.MatchTypeTruePositive
			}

			return godl.MatchTypeFalseNegative
		},
		ValidationObserver: func(confMat godl.ConfusionMatrix, cost float32) {
			fmt.Printf("%v\nCost: %0.4f", confMat, cost)
		},
		CostFn: godl.CategoricalCrossEntropyLoss(godl.CrossEntropyLossOpt{}),
	})
	handleErr("training", err)
}
