export async function uploadImage(file: File): Promise<{result_image_url: string, notes?: string}> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch('/api/process', { method: 'POST', body: fd })
  if (!res.ok) {
    const txt = await res.text().catch(()=> '')
    throw new Error(`Falha ao processar imagem. ${txt}`)
  }
  return res.json()
}
