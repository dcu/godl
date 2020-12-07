package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
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
			desc:          "VBS",
			input:         NewMatrix(g, tensor.Float64, WithShape(100, 50), WithName("input"), WithInit(GlorotN(1.0))),
			vbs:           10,
			expectedShape: tensor.Shape{100, 50},
		},
		{
			desc:          "BiggerVBS",
			input:         NewMatrix(g, tensor.Float64, WithShape(100, 50), WithName("input"), WithInit(GlorotN(1.0))),
			vbs:           128,
			expectedShape: tensor.Shape{100, 50},
		},
		{
			desc:          "FractionVBS",
			input:         NewMatrix(g, tensor.Float64, WithShape(100, 50), WithName("input"), WithInit(GlorotN(1.0))),
			vbs:           4,
			expectedShape: tensor.Shape{100, 50},
		},
	}

	tn := &TabNet{}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			x, err := tn.GBN(tcase.input, GBNOpts{
				VirtualBatchSize: tcase.vbs,
			})

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			}

			c.NoError(err)
			c.Equal(tcase.expectedShape, x.Shape())
		})
	}
}
