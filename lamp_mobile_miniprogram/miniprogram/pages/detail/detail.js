function labelClass(label) {
  if (label === 'positive') return 'positive'
  if (label === 'negative') return 'negative'
  if (label === 'abnormal') return 'abnormal'
  return 'review'
}

function decorateWell(w) {
  return {
    ...w,
    labelClass: labelClass(w.label),
    confidenceText: w.confidence == null ? '-' : String(w.confidence)
  }
}

Page({
  data: {
    result: null,
    wells: [],
    wellNames: [],
    selectedIndex: 0,
    selectedWell: '',
    selectedInfo: null
  },

  onLoad() {
    const result = getApp().globalData.analyzeResult
    if (!result) {
      wx.redirectTo({ url: '/pages/index/index' })
      return
    }
    const wells = (result.wells || []).map(decorateWell)
    const wellNames = wells.map(w => w.well)
    this.setData({
      result,
      wells,
      wellNames,
      selectedIndex: 0,
      selectedWell: wellNames[0] || '',
      selectedInfo: wells[0] || null
    }, () => this.drawCurve())
  },

  onReady() {
    this.drawCurve()
  },

  onSelectWell(e) {
    const idx = Number(e.detail.value)
    this.setSelected(idx)
  },

  tapWell(e) {
    this.setSelected(Number(e.currentTarget.dataset.index))
  },

  setSelected(idx) {
    const wells = this.data.wells
    if (!wells[idx]) return
    this.setData({
      selectedIndex: idx,
      selectedWell: wells[idx].well,
      selectedInfo: wells[idx]
    }, () => this.drawCurve())
  },

  drawCurve() {
    const result = this.data.result
    if (!result || !result.curves) return
    const curve = (result.curves.wells || [])[this.data.selectedIndex]
    const time = result.curves.time || []
    if (!curve || !curve.smooth || !curve.smooth.length) return

    const ctx = wx.createCanvasContext('curveCanvas', this)
    const sys = wx.getSystemInfoSync()
    const width = sys.windowWidth - 56
    const height = 215
    const padL = 42
    const padR = 14
    const padT = 18
    const padB = 36
    const plotW = width - padL - padR
    const plotH = height - padT - padB
    const values = curve.smooth.map(v => Number(v)).filter(v => !Number.isNaN(v))
    if (!values.length) return
    let minY = Math.min(...values)
    let maxY = Math.max(...values)
    if (maxY === minY) maxY = minY + 1
    const span = maxY - minY
    minY -= span * 0.08
    maxY += span * 0.08

    ctx.clearRect(0, 0, width, height)
    ctx.setFillStyle('#fbfcff')
    ctx.fillRect(0, 0, width, height)

    ctx.setStrokeStyle('#e4e8f2')
    ctx.setLineWidth(1)
    for (let i = 0; i <= 4; i++) {
      const y = padT + plotH * i / 4
      ctx.beginPath()
      ctx.moveTo(padL, y)
      ctx.lineTo(width - padR, y)
      ctx.stroke()
    }

    ctx.setStrokeStyle('#9aa3b8')
    ctx.beginPath()
    ctx.moveTo(padL, padT)
    ctx.lineTo(padL, padT + plotH)
    ctx.lineTo(width - padR, padT + plotH)
    ctx.stroke()

    ctx.setFillStyle('#7b8497')
    ctx.setFontSize(10)
    ctx.fillText('荧光', 6, padT + 8)
    ctx.fillText('time', width - 45, height - 10)
    ctx.fillText(String(Math.round(maxY)), 2, padT + 4)
    ctx.fillText(String(Math.round(minY)), 2, padT + plotH)
    ctx.fillText(String(time[0] || 0), padL, height - 10)
    ctx.fillText(String(time[time.length - 1] || values.length), width - 76, height - 10)

    ctx.setStrokeStyle('#3556f5')
    ctx.setLineWidth(2)
    ctx.beginPath()
    curve.smooth.forEach((v, i) => {
      const x = padL + plotW * i / Math.max(curve.smooth.length - 1, 1)
      const y = padT + plotH * (1 - (Number(v) - minY) / (maxY - minY))
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
    ctx.draw()
  }
})
