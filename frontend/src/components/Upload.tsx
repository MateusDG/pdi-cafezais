import { useRef, useState } from 'react'
import { uploadImage } from '../api'

type Props = { onResult: (url: string, notes?: string)=>void }

export default function Upload({ onResult }: Props){
  const inputRef = useRef<HTMLInputElement|null>(null)
  const [busy, setBusy] = useState(false)

  async function onSubmit(e: React.FormEvent){
    e.preventDefault()
    const file = inputRef.current?.files?.[0]
    if(!file) return
    setBusy(true)
    try{
      const data = await uploadImage(file)
      onResult(data.result_image_url, data.notes)
    }catch(err:any){
      alert(err?.message ?? 'Erro')
    }finally{
      setBusy(false)
    }
  }

  return (
    <form onSubmit={onSubmit}>
      <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" required />
      <button className="btn" type="submit" disabled={busy}>{busy ? 'Processando...' : 'Processar'}</button>
    </form>
  )
}
