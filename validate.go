package godl

import (
	"fmt"
	"log"
	"strings"

	"github.com/olekukonko/tablewriter"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MatchType int

const (
	MatchTypeTruePositive MatchType = iota
	MatchTypeTrueNegative
	MatchTypeFalsePositive
	MatchTypeFalseNegative
)

type ConfusionMatrix map[MatchType]int

func (cmat ConfusionMatrix) Accuracy() float64 {
	v := float64(cmat[MatchTypeTruePositive]+cmat[MatchTypeTrueNegative]) / float64(cmat[MatchTypeTrueNegative]+cmat[MatchTypeTruePositive]+cmat[MatchTypeFalseNegative]+cmat[MatchTypeFalsePositive])

	return v
}

func (cmat ConfusionMatrix) Precision() float64 {
	v := float64(cmat[MatchTypeTruePositive]) / float64(cmat[MatchTypeTruePositive]+cmat[MatchTypeFalsePositive])

	return v
}

func (cmat ConfusionMatrix) F1Score() float64 {
	v := float64(2*cmat[MatchTypeTruePositive]) / float64(2*cmat[MatchTypeTruePositive]+cmat[MatchTypeFalsePositive]+cmat[MatchTypeFalseNegative])

	return v
}

func (cmat ConfusionMatrix) Recall() float64 {
	v := float64(cmat[MatchTypeTruePositive]) / float64(cmat[MatchTypeTruePositive]+cmat[MatchTypeFalseNegative])

	return v
}

func (cmat ConfusionMatrix) MissRate() float64 {
	v := 1 - cmat.Recall()

	return v
}

func (cmat ConfusionMatrix) String() string {
	b := strings.Builder{}
	w := tablewriter.NewWriter(&b)
	w.SetBorder(true)
	w.SetAlignment(tablewriter.ALIGN_CENTER)
	w.SetRowLine(true)

	w.SetHeader([]string{"Actual\\Predicted", "P", "N"})
	w.Rich([]string{"P", fmt.Sprintf("%*d", 8, cmat[MatchTypeTruePositive]), fmt.Sprintf("%*d", 8, cmat[MatchTypeFalseNegative])}, []tablewriter.Colors{{}, {tablewriter.BgHiGreenColor, tablewriter.FgBlackColor}, {tablewriter.BgHiRedColor, tablewriter.FgBlackColor}})
	w.Rich([]string{"N", fmt.Sprintf("%*d", 8, cmat[MatchTypeFalsePositive]), fmt.Sprintf("%*d", 8, cmat[MatchTypeTrueNegative])}, []tablewriter.Colors{{}, {tablewriter.BgHiRedColor, tablewriter.FgBlackColor}, {tablewriter.BgHiGreenColor, tablewriter.FgBlackColor}})

	w.Render()

	b.WriteString(fmt.Sprintf(`
Accuracy: %0.1f%%
Precision: %0.1f%%
F1 Score: %0.1f%%
Recall: %0.1f%%
`, cmat.Accuracy()*100, cmat.Precision()*100, cmat.F1Score()*100, cmat.Recall()*100))

	return b.String()
}

func Validate(m *Model, x, y *gorgonia.Node, costVal, predVal gorgonia.Value, validateX, validateY tensor.Tensor, opts TrainOpts) error {
	opts.setDefaults()

	g := m.evalGraph
	if g == nil {
		fatal("evaluation graph not set")
	}

	dl := NewDataLoader(validateX, validateY, DataLoaderOpts{
		BatchSize: opts.BatchSize,
		Shuffle:   false,
	})

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

	vm := gorgonia.NewTapeMachine(g, vmOpts...)

	defer vm.Close()

	confMat := ConfusionMatrix{}

	for dl.HasNext() {
		xVal, yVal := dl.Next()

		err := gorgonia.Let(x, xVal)
		if err != nil {
			fatal("error assigning x: %v", err)
		}

		err = gorgonia.Let(y, yVal)
		if err != nil {
			fatal("error assigning y: %v", err)
		}

		if err = vm.RunAll(); err != nil {
			fatal("Failed batch %d. Error: %v", dl.CurrentBatch, err)
		}

		for j := 0; j < predVal.Shape()[0]; j++ {
			yRowT, err := yVal.Slice(gorgonia.S(j, j+1))
			if err != nil {
				panic(err)
			}

			var yRow []float32

			switch v := yRowT.Data().(type) {
			case []float32:
				yRow = v
			case float32:
				yRow = []float32{v}
			default:
				log.Panicf("type %T not supported", v)
			}

			// get prediction
			predRowT, err := predVal.(tensor.Tensor).Slice(gorgonia.S(j, j+1))
			if err != nil {
				panic(err)
			}

			var predRow []float32

			switch v := predRowT.Data().(type) {
			case []float32:
				predRow = v
			case float32:
				predRow = []float32{v}
			default:
				log.Panicf("type %T not supported", v)
			}

			mt := opts.MatchTypeFor(predRow, yRow)
			confMat[mt]++
		}

		vm.Reset()
	}

	if opts.ValidationObserver != nil {
		opts.ValidationObserver(confMat, costVal.Data().(float32))
	}

	return nil
}
