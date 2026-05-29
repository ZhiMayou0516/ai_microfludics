function labelClass(label) {
  if (label === 'positive') return 'positive'
  if (label === 'negative') return 'negative'
  if (label === 'abnormal') return 'abnormal'
  return 'review'
}

function iconFor(label) {
  if (label === 'positive') return '＋'
  if (label === 'negative') return '－'
  if (label === 'abnormal') return '!'
  return '?'
}

Page({
  data: {
    result: null,
    overallClass: 'review',
    statusIcon: '?',
    previewWells: []
  },

  onLoad() {
    const result = getApp().globalData.analyzeResult
    if (!result) {
      wx.redirectTo({ url: '/pages/index/index' })
      return
    }
    const overallClass = labelClass(result.overall_result)
    const previewWells = (result.wells || []).map(w => ({
      ...w,
      confidence: w.confidence == null ? '-' : w.confidence,
      previewText: w.reason || ('置信度 ' + (w.confidence == null ? '-' : w.confidence)),
      labelClass: labelClass(w.label)
    }))
    this.setData({
      result,
      overallClass,
      statusIcon: iconFor(result.overall_result),
      previewWells
    })
  },

  goDetail() {
    wx.navigateTo({ url: '/pages/detail/detail' })
  },

  backHome() {
    wx.reLaunch({ url: '/pages/index/index' })
  }
})
