"""Native dialogue planner: QA never enters memory; media materialization is explicit."""
import base64
import hashlib
import json
from pathlib import PurePosixPath
from .protocol import canonical,validate_add


def media_reference(path):
    if not isinstance(path,str) or not path.startswith('../image/') or '\\' in path:
        raise ValueError('unrecognized native image reference')
    relative=PurePosixPath(path[3:])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='image':
        raise ValueError('native image escapes approved data root')
    return relative.as_posix()


def plan_rounds(container,namespace):
    """One source round per Add, original session grouping and both role strings.

    Reads only multi_session_dialogues. Native profile and QAs remain untouched in
    the quarantined container. Source date stays in the provenance sidecar: no
    timezone is invented. A planned image_ref is NOT an AML protocol content part.
    """
    if not isinstance(namespace,str) or not namespace:raise ValueError('explicit dataset/task namespace required')
    sessions=container.get('multi_session_dialogues')
    if not isinstance(sessions,list):raise ValueError('native multi-session array required')
    plans=[];seen=set()
    for session in sessions:
        sid=session['session_id'];date=session['date']
        if not isinstance(sid,str) or not sid or not isinstance(date,str):raise ValueError('native session/date required')
        for item in session['dialogues']:
            rid=item['round'];key=(sid,rid)
            if key in seen:raise ValueError('duplicate native session/round')
            seen.add(key)
            if not isinstance(rid,str) or not rid:raise ValueError('native round required')
            user=item['user'];assistant=item['assistant']
            if any(not isinstance(value,str) or not value.strip() for value in (user,assistant)):
                raise ValueError('original nonempty role strings required')
            images=item.get('input_image',[]);captions=item.get('image_caption',[]);ids=item.get('image_id',[])
            if any(not isinstance(values,list) for values in (images,captions,ids)) or not len(images)==len(captions)==len(ids):
                raise ValueError('native image/caption/id arrays must align; no silent first-image truncation')
            parts=[dict(type='text',text=user)];occurrences=[]
            for index,(path,caption,image_id) in enumerate(zip(images,captions,ids)):
                if not isinstance(caption,str):raise ValueError('original image caption must be text')
                relative=media_reference(path);part_ordinal=len(parts)
                parts.append(dict(type='native_image_ref',relative_path=relative))
                if caption.strip():parts.append(dict(type='text',text=caption))
                occurrences.append(dict(image_id=image_id,image_index=index,part_ordinal=part_ordinal,relative_path=relative))
            digest=hashlib.sha256(canonical([namespace,sid,rid]).encode()).hexdigest()
            plans.append(dict(request_id='native-'+digest,user_id=namespace,session_id=namespace+':'+sid,
                messages=[dict(role='user',content=parts),dict(role='assistant',content=assistant)],
                provenance=dict(source_session=sid,source_round=rid,source_date=date,images=occurrences)))
    return plans


def materialize_round(plan,media_reader):
    """The callback must enforce reviewed path/hash/license. Never fetch remote URLs."""
    if not callable(media_reader):raise ValueError('explicit verified local evidence reader required')
    payload=json.loads(json.dumps({key:plan[key] for key in ('request_id','user_id','session_id','messages')}))
    for message in payload['messages']:
        if not isinstance(message['content'],list):continue
        for index,part in enumerate(message['content']):
            if part['type']!='native_image_ref':continue
            raw=media_reader(part['relative_path'])
            if not isinstance(raw,bytes):raise ValueError('local image reader must return bytes')
            mime=('png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else 'jpeg' if raw.startswith(b'\xff\xd8\xff')
                  else 'webp' if raw.startswith(b'RIFF') and raw[8:12]==b'WEBP' else None)
            if mime is None:raise ValueError('unsupported native image signature')
            message['content'][index]=dict(type='image_url',image_url=dict(url='data:image/'+mime+';base64,'+base64.b64encode(raw).decode('ascii')))
    validate_add(payload)
    return payload
