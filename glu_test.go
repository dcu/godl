package tabnet

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func initDummyWeights(dt tensor.Dtype, s ...int) interface{} {
	v := make([]float64, tensor.Shape(s).TotalSize())
	rs := rand.New(rand.NewSource(0)) // fixed rand source, will generate same numbers in same order

	for i := range v {
		v[i] = rs.Float64()
	}

	return v
}

func TestGLU(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc           string
		input          *Node
		vbs            int
		output         int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float64
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(10, 1), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(10, 1),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
				),
			)),
			vbs:            2,
			output:         5,
			expectedShape:  tensor.Shape{10, 5},
			expectedOutput: []float64{-0.02032319658344429, -0.23820033421818848, -0.1498249294410549, -0.4695500772473174, -0.19614948821642125, 1.209451037512029, -0.06425768971083588, 1.07166679797495, -0.7708185358792397, 0.115167376985537, 0.3628939974383521, -0.22195628192208472, 0.17493446364420168, -0.9165652942191849, -0.12372842821227946, 0.8715751287877753, -0.16794957860849727, 0.7509382961623241, -1.0604181328632352, -0.01580335449683695, 0.4617282591649743, -0.2150087375277546, 0.28464243871608585, -0.992893543185774, -0.10502838524171738, 0.7711280592383527, -0.18333146087666177, 0.640526722956774, -1.0802843882129483, -0.04074970479756943, 0.5053822930536606, -0.21154795277041097, 0.33458982107350616, -1.0191526021331747, -0.09662151715592984, 0.726766946660376, -0.18911671373955366, 0.5903732163146284, -1.0816180936572668, -0.05090197188691691, 0.5298286856221519, -0.20949303270316708, 0.36279576842531835, -1.0318305172540285, -0.09185465342897303, 0.7019963686964089, -0.19212374868317392, 0.5620820214823424, -1.0803836725945108, -0.056386870039618876},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			x, err := tn.GLU(GLUOpts{
				VirtualBatchSize: tcase.vbs,
				OutputDimension:   tcase.output,
				WeightsInit:      initDummyWeights,
			})(tcase.input)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float64))
		})
	}
}
