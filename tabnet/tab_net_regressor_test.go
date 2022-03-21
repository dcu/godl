package tabnet

import (
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestTabNetRegressor(t *testing.T) {
	testCases := []struct {
		desc              string
		epochs            int
		input             tensor.Tensor
		target            tensor.Tensor
		vbs               int
		independentBlocks int
		sharedBlocks      int
		output            int
		steps             int
		gamma             float64
		prediction        int
		attentive         int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float32
		expectedGrad      []float32
		expectedCost      float64
		expectedAcumLoss  float64
	}{
		{
			desc:   "Example 1",
			epochs: 1,
			input: tensor.New(
				tensor.WithShape(4, 4),
				tensor.WithBacking([]float32{0.4, 1.4, 2.4, 0, 4.4, 5.4, 6.4, 1, 8.4, 9.4, 10.4, 2, 12.4, 13.4, 14.4, 3}),
			),
			target: tensor.New(
				tensor.WithShape(4, 1),
				tensor.WithBacking([]float32{1, 1, 0, 0}),
			),
			vbs:               128,
			output:            1,
			independentBlocks: 2,
			sharedBlocks:      2,
			steps:             3,
			gamma:             1.3,
			prediction:        8,
			attentive:         8,
			expectedShape:     tensor.Shape{4, 12},
			expectedOutput:    []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882, 447.8014308162882},
			expectedGrad:      []float32{0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332, 0.020833333333333332},
			expectedCost:      223.90071540814404,
			expectedAcumLoss:  -1.6094379119341007,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			regressor := NewRegressor(tcase.input.Shape()[1], []int{4}, []int{3}, []int{2}, RegressorOpts{
				VirtualBatchSize:   tcase.vbs,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: tcase.prediction,
				AttentionLayerDim:  tcase.attentive,
				SharedBlocks:       tcase.sharedBlocks,
				DecisionSteps:      tcase.steps,
				Gamma:              tcase.gamma,
				BatchSize:          tcase.input.Shape()[0],
				WeightsInit:        initDummyWeights,
				ScaleInit:          gorgonia.Ones(),
				BiasInit:           gorgonia.Zeroes(),
				Epsilon:            1e-15,
				Momentum:           0.02,
				WithBias:           false,
			})

			err := regressor.Train(tcase.input, tcase.target, tcase.input, tcase.target, godl.TrainOpts{
				Epochs:    tcase.epochs,
				BatchSize: tcase.input.Shape()[0],
				DevMode:   true,
				Solver:    gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.02), gorgonia.WithClip(1.0)),
				MatchTypeFor: func(predVal, targetVal []float32) godl.MatchType {
					t.Logf("%v vs %v", predVal, targetVal)

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
					t.Logf("%v\nCost: %0.4f", confMat, cost)
				},
			})
			c.NoError(err)

			regressor.model.PrintWatchables()

			for _, n := range regressor.model.Learnables() {
				t.Logf("%s: %v", n.Name(), n.Value().Data().([]float32)[0:2])
			}

			// y := result.Output

			// if tcase.expectedErr != "" {
			// 	c.Error(err)

			// 	c.Equal(tcase.expectedErr, err.Error())

			// 	return
			// } else {
			// 	c.NoError(err)
			// }

			// cost := gorgonia.Must(gorgonia.Mean(y))
			// _, err = gorgonia.Grad(cost, append([]*gorgonia.Node{x}, tn.Learnables()...)...)
			// c.NoError(err)

			// vm := gorgonia.NewTapeMachine(g,
			// 	gorgonia.BindDualValues(tn.Learnables()...),
			// 	gorgonia.WithLogger(testLogger),
			// 	gorgonia.WithValueFmt("%+v"),
			// 	gorgonia.WithWatchlist(),
			// 	gorgonia.WithNaNWatch(),
			// 	gorgonia.WithInfWatch(),
			// )
			// c.NoError(vm.RunAll())

			// tn.PrintWatchables()
			// // fmt.Printf("%v\n", g.String())

			// log.Printf("input grad: %v", x.Deriv().Value())

			// c.Equal(tcase.expectedShape, y.Shape())

			// log.Printf("y: %#v", y.Value().Data())
			// c.InDeltaSlice(tcase.expectedOutput, y.Value().Data().([]float32), 1e-5)

			// yGrad, err := y.Grad()
			// c.NoError(err)

			// c.Equal(tcase.expectedGrad, yGrad.Data())
			// c.InDelta(tcase.expectedCost, cost.Value().Data(), 1e-5)
			// c.Equal(tcase.expectedAcumLoss, result.Loss.Value().Data())

			// weightsByName := map[string]*gorgonia.Node{}

			// for _, n := range tn.Learnables() {
			// 	weightsByName[n.Name()] = n

			// 	wGrad, err := n.Grad()
			// 	c.NoError(err)
			// 	log.Printf("%s: %v", n.Name(), wGrad.Data().([]float32)[0:2])
			// }

			// // {
			// // 	w := weightsByName["BatchNorm1d_31.81.scale.1.5"]
			// // 	wGrad, err := w.Grad()
			// // 	c.NoError(err)
			// // 	c.Equal([]float32{0.0024, 0.0024, 0.0024, 0.0000, 0.0000}, wGrad.Data(), w.Name())
			// // }

			// optim := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.02))
			// err = optim.Step(gorgonia.NodesToValueGrads(tn.Learnables()))
			// c.NoError(err)

			// {
			// 	w := weightsByName["BatchNorm1d_1.1.scale.1.5"]
			// 	log.Printf("weight updated: %v\n\n\n", w.Value())
			// 	c.Equal([]float32{0.9800000823404622, 0.9800000823404622, 0.9800000823404622, 1, 1}, w.Value().Data(), w.Name())
			// }

			// for _, n := range tn.Learnables() {
			// 	log.Printf("%s: %v", n.Name(), n.Value().Data().([]float32)[0:2])
			// }
		})
	}
}
