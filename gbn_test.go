package tabnet

import (
	"fmt"
	"log"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGBN(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc          string
		input         *Node
		vbs           int
		expectedShape tensor.Shape
		expectedErr   string
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(1, 10), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(1, 10),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
				),
			)),
			vbs:           5,
			expectedShape: tensor.Shape{1, 10},
		},
		// {
		// 	desc:          "BiggerVBS",
		// 	input:         NewMatrix(g, tensor.Float64, WithShape(100, 50), WithName("input"), WithInit(GlorotN(1.0))),
		// 	vbs:           128,
		// 	expectedShape: tensor.Shape{100, 50},
		// },
		// {
		// 	desc:          "FractionVBS",
		// 	input:         NewMatrix(g, tensor.Float64, WithShape(100, 50), WithName("input"), WithInit(GlorotN(1.0))),
		// 	vbs:           4,
		// 	expectedShape: tensor.Shape{100, 50},
		// 	expectedErr:   "shape size doesn't not match. Expected 4800, got 5000",
		// },
	}

	tn := &TabNet{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			y, err := tn.GBN(tcase.input, GBNOpts{
				VirtualBatchSize: tcase.vbs,
			})

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			}

			c.NoError(err)
			c.Equal(tcase.expectedShape, y.Shape())

			vm := NewTapeMachine(tn.g,
				gorgonia.WithLogger(log.New(os.Stderr, "[gorgonia]", log.LstdFlags)),
				gorgonia.BindDualValues(tn.learnables...),
				// gorgonia.WithValueFmt("%+v"),
				// gorgonia.WithWatchlist(),
			)
			c.NoError(vm.RunAll())

			fmt.Printf("output: %v\n%v\n", y.Value(), y.Shape())
		})
	}
}
