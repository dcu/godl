package table

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"gorgonia.org/tensor"
)

type Cell struct {
	Dtype tensor.Dtype
	V     any
}

func (v Cell) Int() int {
	return v.V.(int)
}

func (v Cell) Float32() float32 {
	return v.V.(float32)
}

func (v Cell) String() string {
	return fmt.Sprintf("%v", v.V)
}

func StringToCell(v string) *Cell {
	i, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
	if err == nil {
		return &Cell{tensor.Int, int(i)}
	}

	f, err := strconv.ParseFloat(strings.TrimSpace(v), 32)
	if err == nil {
		return &Cell{tensor.Float32, float32(f)}
	}

	return &Cell{tensor.String, v}
}

type Row struct {
	Cells []*Cell
	Tags  map[string]bool
}

func (r *Row) AddTag(tags ...string) {
	for _, tag := range tags {
		r.Tags[tag] = true
	}
}

func (r Row) HasAnyTag(tags []string) bool {
	if len(tags) == 0 {
		return true
	}

	for _, tag := range tags {
		if r.Tags[tag] {
			return true
		}
	}

	return false
}

func StringsToRow(values []string) Row {
	cells := make([]*Cell, len(values))
	for i, v := range values {
		cells[i] = StringToCell(v)
	}

	return Row{cells, map[string]bool{}}
}

type Rows []*Row

type Table struct {
	Header []string
	Rows   Rows

	ClassesByColumn map[int][]string
}

// ReadCSV loads a CSV table
func ReadCSV(pathCSV string) (*Table, error) {
	f, err := os.Open(pathCSV)
	if err != nil {
		return nil, err
	}

	defer func() { _ = f.Close() }()

	t := &Table{
		ClassesByColumn: map[int][]string{},
	}

	knownClasses := map[int]map[string]bool{}

	csvReader := csv.NewReader(f)
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}

		if t.Header == nil {
			t.Header = record

			for i := range t.Header {
				knownClasses[i] = map[string]bool{}
			}
		}

		if err != nil {
			return nil, err
		}

		cells := make([]*Cell, len(record))
		for i, r := range record {
			v := StringToCell(r)

			cells[i] = v

			if v.Dtype == tensor.String && !knownClasses[i][v.V.(string)] {
				t.ClassesByColumn[i] = append(t.ClassesByColumn[i], v.V.(string))

				knownClasses[i][v.V.(string)] = true
			}
		}

		t.Rows = append(t.Rows, &Row{cells, map[string]bool{}})
	}

	return t, nil
}

func (t *Table) Has(column string) bool {
	for _, n := range t.Header {
		if n == column {
			return true
		}
	}

	return false
}

func (t *Table) AddColumn(columnName string, val interface{}) {
	switch v := val.(type) {
	case func() string:
		{
			for i := range t.Rows {
				t.Rows[i].Cells = append(t.Rows[i].Cells, StringToCell(v()))
			}
		}
	default:
		for i := range t.Rows {
			t.Rows[i].Cells = append(t.Rows[i].Cells, StringToCell(fmt.Sprintf("%v", val)))
		}
	}
}

func (t *Table) AddTag(tagFunc func() string) {
	for i := range t.Rows {
		tag := tagFunc()
		if tag != "" {
			t.Rows[i].Tags[tag] = true
		}
	}
}

func (t *Table) EachColumn(f func(columnName string, v *Cell)) {
	if len(t.Rows) == 0 {
		return
	}

	for i, h := range t.Rows[0].Cells {
		f(t.Header[i], h)
	}
}

func (t *Table) EachRow(f func(row *Row)) {
	if len(t.Rows) == 0 {
		return
	}

	for _, row := range t.Rows {
		f(row)
	}
}

func (t *Table) EachCell(cb func(rowNumber, columnNumber int, cell *Cell)) {
	for rowNumber, row := range t.Rows {
		for columnNumber, cell := range row.Cells {
			cb(rowNumber, columnNumber, cell)
		}
	}
}

type ToTensorOpts struct {
	TargetColumns []int
	SelectTags    []string
}

func (opt *ToTensorOpts) setDefaults() {
	if opt.TargetColumns == nil {
		opt.TargetColumns = []int{}
	}
}

func (t *Table) ToTensors(opts ToTensorOpts) (x *tensor.Dense, y *tensor.Dense) {
	opts.setDefaults()

	indexes := make(map[int]map[string]int, len(t.ClassesByColumn))

	for col, cats := range t.ClassesByColumn {
		sort.Strings(cats)

		indexes[col] = make(map[string]int, len(cats))
		for i, c := range cats {
			indexes[col][c] = i
		}
	}

	targetColumnsIndexed := make(map[int]bool, len(opts.TargetColumns))
	for _, col := range opts.TargetColumns {
		targetColumnsIndexed[col] = true
	}

	width := len(t.Header) - len(opts.TargetColumns)
	backing := make([]float32, 0, width*len(t.Rows))
	targetBacking := make([]float32, 0, len(t.Rows)*len(opts.TargetColumns))

	var rowsCount int

	for _, row := range t.Rows {
		include := row.HasAnyTag(opts.SelectTags)
		if !include {
			continue
		}

		for colIndex, cell := range row.Cells {
			var val float32

			if m, ok := indexes[colIndex]; ok {
				// this is a categorical column which is encoded as a number
				val = float32(m[fmt.Sprintf("%v", cell.V)])
			} else if cell.Dtype == tensor.Int {
				val = float32(cell.V.(int))
			} else if cell.Dtype == tensor.Float64 {
				val = float32(cell.V.(float64))
			} else if cell.Dtype == tensor.Float32 {
				val = cell.V.(float32)
			} else {
				log.Panicf("unsupported type: %v", cell.Dtype)
			}

			if targetColumnsIndexed[colIndex] {
				targetBacking = append(targetBacking, val)
			} else {
				backing = append(backing, val)
			}
		}

		rowsCount++
	}

	x = tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(rowsCount, width),
		tensor.WithBacking(backing),
	)

	if len(opts.TargetColumns) > 0 {
		y = tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(rowsCount, len(opts.TargetColumns)),
			tensor.WithBacking(targetBacking),
		)
	}

	return x, y
}

func (t *Table) CategoricalColumns(excludeColumns ...int) (columns []int, dimensions []int) {
	for col := range t.ClassesByColumn {
		if !isIn(col, excludeColumns) {
			columns = append(columns, col)
		}
	}

	sort.Ints(columns)

	for _, col := range columns {
		dimensions = append(dimensions, len(t.ClassesByColumn[col]))
	}

	return columns, dimensions
}

func RandValueIn(valueAndProbability map[string]float64) func() string {
	totalProb := 0.0

	values := make([]string, 0, len(valueAndProbability))
	thresholds := make([]float64, 0, len(valueAndProbability))

	for val, prob := range valueAndProbability {
		totalProb += prob

		thresholds = append(thresholds, totalProb)
		values = append(values, val)
	}

	if totalProb != 1 {
		log.Panicf("probabilities sum must be 1")
	}

	return func() string {
		r := rand.Float64()

		for i, p := range thresholds {
			if r <= p {
				return values[i]
			}
		}

		// this can't happen
		return ""
	}
}

func isIn(x int, a []int) bool {
	for _, v := range a {
		if x == v {
			return true
		}
	}

	return false
}

var (
	_ fmt.Stringer = Cell{}
)
