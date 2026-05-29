const { uploadAnalyze } = require('../../utils/api')

Page({
  data: {
    filePath: '',
    fileName: '',
    sampleId: '',
    mode: 'rule',
    dropFirstRows: 5,
    smoothWindow: 5,
    dropTimeColumn: false,
    showAdvanced: false,
    loading: false
  },

  chooseFile() {
    wx.chooseMessageFile({
      count: 1,
      type: 'file',
      extension: ['csv', 'xlsx', 'xls'],
      success: (res) => {
        const file = res.tempFiles[0]
        this.setData({ filePath: file.path, fileName: file.name })
      }
    })
  },

  onSampleInput(e) {
    this.setData({ sampleId: e.detail.value })
  },

  setMode(e) {
    this.setData({ mode: e.currentTarget.dataset.mode })
  },

  toggleAdvanced() {
    this.setData({ showAdvanced: !this.data.showAdvanced })
  },

  onDropRowsInput(e) {
    this.setData({ dropFirstRows: Number(e.detail.value || 0) })
  },

  onSmoothInput(e) {
    let value = Number(e.detail.value || 1)
    if (value < 1) value = 1
    this.setData({ smoothWindow: value })
  },

  onTimeColumnChange(e) {
    this.setData({ dropTimeColumn: e.detail.value })
  },

  async startAnalyze() {
    if (!this.data.filePath) {
      wx.showToast({ title: '请先上传文件', icon: 'none' })
      return
    }
    this.setData({ loading: true })
    try {
      const result = await uploadAnalyze({
        filePath: this.data.filePath,
        fileName: this.data.fileName,
        sampleId: this.data.sampleId,
        mode: this.data.mode,
        dropFirstRows: this.data.dropFirstRows,
        smoothWindow: this.data.smoothWindow,
        dropTimeColumn: this.data.dropTimeColumn
      })
      getApp().globalData.analyzeResult = result
      wx.navigateTo({ url: '/pages/result/result' })
    } catch (err) {
      wx.showModal({ title: '检测失败', content: err.message || String(err), showCancel: false })
    } finally {
      this.setData({ loading: false })
    }
  }
})
