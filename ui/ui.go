package ui

import (
	"context"
	"fmt"
	"time"

	"github.com/mum4k/termdash"
	"github.com/mum4k/termdash/cell"
	"github.com/mum4k/termdash/container"
	"github.com/mum4k/termdash/keyboard"
	"github.com/mum4k/termdash/linestyle"
	"github.com/mum4k/termdash/terminal/termbox"
	"github.com/mum4k/termdash/terminal/terminalapi"
	"github.com/mum4k/termdash/widgets/gauge"
	"github.com/mum4k/termdash/widgets/linechart"
	"github.com/mum4k/termdash/widgets/text"
)

const (
	rootID = "root"
)

type UI struct {
	term                  *termbox.Terminal
	container             *container.Container
	epochsBar, batchesBar *gauge.Gauge
	costLine              *linechart.LineChart
	costText              *text.Text
	costs                 []float64
}

func New() *UI {
	t, err := termbox.New(termbox.ColorMode(terminalapi.ColorMode256))
	if err != nil {
		panic(err)
	}

	epochsBar, err := gauge.New(
		gauge.TextLabel("Epoch"),
		gauge.Color(cell.ColorCyan),
		gauge.FilledTextColor(cell.ColorBlack),
		gauge.EmptyTextColor(cell.ColorYellow),
	)
	if err != nil {
		panic(err)
	}

	batchesBar, err := gauge.New(
		gauge.TextLabel("Batch"),
		gauge.Color(cell.ColorCyan),
		gauge.FilledTextColor(cell.ColorBlack),
		gauge.EmptyTextColor(cell.ColorYellow),
	)
	if err != nil {
		panic(err)
	}

	costLine, err := linechart.New(
		linechart.AxesCellOpts(cell.FgColor(cell.ColorCyan)),
		linechart.YLabelCellOpts(cell.FgColor(cell.ColorGreen)),
		linechart.XLabelCellOpts(cell.FgColor(cell.ColorGreen)),
	)
	if err != nil {
		panic(err)
	}

	costText, err := text.New(text.RollContent())
	if err != nil {
		panic(err)
	}

	c, err := container.New(
		t, container.ID(rootID),
		container.SplitVertical(
			container.Left(
				container.SplitHorizontal(
					container.Top(
						container.Border(linestyle.Light),
						container.PlaceWidget(costLine),
					),
					container.Bottom(
						container.PlaceWidget(costText),
						container.Border(linestyle.Light),
					),
				),
			),
			container.Right(
				container.SplitHorizontal(
					container.Top(
						container.PlaceWidget(epochsBar),
						container.Border(linestyle.Light),
					),
					container.Bottom(
						container.PlaceWidget(batchesBar),
						container.Border(linestyle.Light),
					),
				),
			),
		),
	)
	if err != nil {
		panic(err)
	}

	return &UI{
		term:       t,
		container:  c,
		epochsBar:  epochsBar,
		batchesBar: batchesBar,
		costLine:   costLine,
		costText:   costText,
	}
}

func (ui *UI) Start() {
	ctx, cancel := context.WithCancel(context.Background())

	quitter := func(k *terminalapi.Keyboard) {
		if k.Key == keyboard.KeyEsc || k.Key == keyboard.KeyCtrlC {
			cancel()
		}
	}
	if err := termdash.Run(ctx, ui.term, ui.container, termdash.KeyboardSubscriber(quitter), termdash.RedrawInterval(250*time.Millisecond)); err != nil {
		panic(err)
	}
}

func (ui *UI) UpdateCost(epoch, epochs int, batch, batches int, cost float64) {
	_ = ui.epochsBar.Percent(int(100 * (float64(epoch) / float64(epochs))))
	_ = ui.batchesBar.Percent(int(100 * (float64(batch) / float64(batches))))

	_ = ui.costText.Write(fmt.Sprintf("%d/%d cost: %v\n", epoch, epochs, cost))

	ui.costs = append(ui.costs, float64(cost))
	if len(ui.costs) > 100 {
		ui.costs = ui.costs[len(ui.costs)-100:]
	}

	_ = ui.costLine.Series("cost", ui.costs)
}
