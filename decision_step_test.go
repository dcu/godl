package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestDecisionStep(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc              string
		input             *Node
		vbs               int
		independentBlocks int
		output            int
		expectedShape     tensor.Shape
		expectedErr       string
		expectedOutput    []float64
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(12, 1), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(12, 1),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4}),
				),
			)),
			vbs:               2,
			output:            2,
			independentBlocks: 5,
			expectedShape:     tensor.Shape{12, 11},
			expectedOutput:    []float64{1.4203781691538828, -1.266146834590152, 0.699123605376538, -1.3790340326984967, -0.3809099740371106, 0.5395496579500985, -1.1894226382453932, 0.5415199795647168, -1.0186457023172237, 1.1082721065532284, -0.7227549161437159, 0.909290354995071, -0.6233033323512835, -0.1936224119192922, -0.1717461748298142, -0.5693763724123883, 0.34333808511124664, -1.0045975811564394, 0.4804384158382873, -1.2712281848920894, 0.8299079831911982, -1.0415862402944884, 1.464517401851036, -1.2619889057799552, 0.7377723962252706, -1.3715671192631997, -0.34698247274036376, 0.521692704087314, -1.2067249156498037, 0.5920497088532225, -1.0753626556518092, 0.9752019709903974, -0.8946977284546624, -1.0756317947176304, 0.5849983355901338, -0.891896168296308, 0.8625860854984944, -0.25460978922718824, 0.24131227361322355, 0.7593603183204046, -0.16780591122988156, -1.258593088228693, 0.438404516716199, -0.39863251545366063, 1.4765073658899082, -1.2579776492945154, 0.7532601863580721, -1.371293924421703, -0.3365262674247697, 0.35803122927218367, -1.3707540125835833, 0.6093298002186751, -1.3685716450603989, 0.8366725415444134, -1.1441853641757038, -1.0806000171276753, 0.6006510317371262, -0.8811193346005317, 0.8626760472676054, -0.24228414680743426, 0.2508765766951338, 0.7592115055392736, -0.16772712802711323, -1.247194526455915, 0.438999369308796, -0.38958810209590533, 1.4876749753625296, -1.2568141185625787, 0.7654627913013393, -1.3712096283396296, -0.32660837038524737, 0.3589703971425325, -1.3698999286718152, 0.6249474850053797, -1.357652724961858, 0.7418923471515086, -1.2425824613754002, -1.0761964220654323, 0.6071117876357346, -0.870246117776577, 0.8627054307236279, -0.23113404922529204, 0.25486219743170446, 0.7530227349178192, -0.167813597864703, -1.236057422924836, 0.43920428571688125, -0.3796206806659796, 1.498694015396826, -1.2562153131896157, 0.7770280478740266, -1.3711331744449327, -0.31681466139567627, 0.4725552188673998, -1.2563448097040462, 0.6398164055197336, -1.3464924439220962, 0.8983258414650196, -1.088556148802897, -1.0683480368136178, 0.6102532345598196, -0.8593502778483342, 0.8627187792809373, -0.22016056860801297, 0.1559137749875454, 0.6438827911002595, -0.16954692591375975, -1.22341484825608, 0.43927735981092825, -0.36911561433339324, 1.5096663196400086, -1.2559601878075168, 0.7882273897396113, -1.3709778720042394, -0.30714128263190515, 0.4732688477146908, -1.2556440129749151, 0.6540726656526616, -1.33253598192282, 0.8661851150956432, -1.1208566138773837, 0.894921180943618, -0.5538067446437659, 0.5888694295173087, -0.9254837866898158, -0.23341534723674826, 0.8884873867931339, -0.4013346220696292, 0.4771990997282917, -1.2100448751812236, 0.42384775489047904, -1.3775723749364266},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			a := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			priors := NewTensor(g, Float64, tcase.input.Dims(), WithShape(tcase.input.Shape()...), WithInit(Ones()))
			x, err := tn.DecisionStep(DecisionStepOpts{
				VirtualBatchSize:   tcase.vbs,
				Shared:             nil,
				IndependentBlocks:  tcase.independentBlocks,
				PredictionLayerDim: 10,
				AttentionLayerDim:  1,
				WeightsInit:        initDummyWeights,
			})(tcase.input, a, priors)

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